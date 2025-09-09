import numpy as np
import pandas as pd
import difflib
from pathlib import Path
from fastapi import HTTPException

DATA_DIR = Path(__file__).parent / "data"
PARFUMS_CSV = DATA_DIR / "parfums_enrichi.csv"

CANDIDATE_FAM_COLS = [
    "famille", "familles", "famille olfactive", "familles olfactives",
    "famille_olfactive", "familles_olfactives",
    "olfactive_family", "olfactory_family", "family", "families"
]

def _find_fam_col(df: pd.DataFrame) -> str | None:
    norm = {c.lower().strip(): c for c in df.columns}
    for key in CANDIDATE_FAM_COLS:
        if key in norm:
            return norm[key]
    return None

def _read_csv_safely(path: Path) -> pd.DataFrame:
    # 1) lecture standard ; header à la ligne 0
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=";", encoding=enc, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            fam = _find_fam_col(df)
            if fam:
                return df
            break
        except UnicodeDecodeError:
            continue

    # 2) si pas trouvé, retenter en supposant une ligne parasite -> header=1
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=";", header=1, encoding=enc, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            fam = _find_fam_col(df)
            if fam:
                return df
        except UnicodeDecodeError:
            continue

    # 3) dernier recours: header inconnu -> on cherche la ligne d’entêtes dans les 3 premières lignes
    df0 = pd.read_csv(path, sep=";", header=None, engine="python", encoding="latin-1")
    for r in range(min(3, len(df0))):
        header = [str(x).strip() for x in df0.iloc[r].tolist()]
        norm = [h.lower() for h in header]
        if any(x in norm for x in CANDIDATE_FAM_COLS):
            df = df0.iloc[r+1:].copy()
            df.columns = header
            df.columns = [str(c).strip() for c in df.columns]
            fam = _find_fam_col(df)
            if fam:
                return df

    raise HTTPException(
        status_code=500,
        detail=f"Aucune colonne de familles olfactives trouvée. Colonnes lues: {list(df0.columns)} (mode header=None)."
    )

def charger_parfums_df() -> tuple[pd.DataFrame, str]:
    df = _read_csv_safely(PARFUMS_CSV)
    fam_col = _find_fam_col(df)
    if not fam_col:
        raise HTTPException(status_code=500, detail=f"Aucune colonne 'famille' détectée. Colonnes: {list(df.columns)}")
    # (debug utile)
    print(f"[parfums_enrichi.csv] colonnes={list(df.columns)} ; fam_col='{fam_col}' ; n={len(df)}")
    return df, fam_col

def get_u_final(u_vector):
    print("📥 u_vector (input):", u_vector)

    # Chargement de la matrice de similarité
    # APRÈS
S_df = charger_matrice_similarite()
    S = S_df.to_numpy()
    u = np.array(u_vector)

    print("✅ Matrice S (shape):", S.shape)
    print("✅ Vecteur u (shape):", u.shape)

    # Vérification de compatibilité
    if S.shape[1] != u.shape[0]:
        raise ValueError(f"Incompatibilité dimensions: S.shape={S.shape}, u.shape={u.shape}")

    return S @ u

def calculate_similarities(u_final):
    # Chargement du fichier des parfums enrichis
   df_parfums, fam_col = charger_parfums_df()

    # Nettoyage des colonnes
    df_parfums.columns = df_parfums.columns.str.strip().str.replace('\u202f|\u00a0', '', regex=True)

    # Renommage des colonnes utiles
    df_parfums.rename(columns={
        'ï»¿images parfums': 'Image',
        'Lien de redirection': 'URL'
    }, inplace=True)

    # Familles olfactives à matcher
    familles_olfactives = [
        'Epicee', 'Ambree', 'Boisee Mousse', 'Hesperidee', 'Florale', 'Aromatique',
        'Cuir', 'Boisee', 'Balsamique', 'Florale Fraiche', 'Verte', 'Florale Rosee',
        'Musquee', 'Fruitee', 'Florale Poudree', 'Marine', "Fleur D'Oranger",
        'Conifere Terpenique', 'Aldehydee'
    ]

    # Recherche des colonnes correspondantes
    correspondance = {}
    colonnes_fichier = df_parfums.columns.tolist()
    for famille in familles_olfactives:
        match = difflib.get_close_matches(famille, colonnes_fichier, n=1, cutoff=0.6)
        if match:
            correspondance[famille] = match[0]

    # Sélection des colonnes notes
    note_columns = df_parfums[[v for v in correspondance.values()]]
    ufinal = u_final[:note_columns.shape[1]]

    # Calcul de la similarité cosinus
    def cosine_similarity(v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return 0 if norm1 == 0 or norm2 == 0 else np.dot(v1, v2) / (norm1 * norm2)

    similarities = []
    for i, row in note_columns.iterrows():
        sim = cosine_similarity(row.values.astype(float), ufinal)
        parfum_name = f"{df_parfums.loc[i, 'Marque']} - {df_parfums.loc[i, 'Nom du Parfum']}"
        image = df_parfums.loc[i, 'Image'] if 'Image' in df_parfums.columns else ""
        url = df_parfums.loc[i, 'URL'] if 'URL' in df_parfums.columns else ""

        similarities.append({
            "parfum": parfum_name,
            "similarité": round(sim * 100, 2),
            "image": image,
            "url": url
        })

    # Tri décroissant
    similarities.sort(key=lambda x: x["similarité"], reverse=True)
    return similarities[:5]

