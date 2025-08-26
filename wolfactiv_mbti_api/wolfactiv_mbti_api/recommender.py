import numpy as np
import pandas as pd
import difflib

def get_u_final(u_vector):
    print("📥 u_vector (input):", u_vector)

    # Chargement de la matrice de similarité
    S_df = pd.read_csv(
        "C:/Users/helen/Downloads/wolfactiv_backend_complet/data/similarité matrice.csv",
        index_col=0,
        encoding="ISO-8859-1", 
        sep=";"
    )
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
    df_parfums = pd.read_csv(
        "C:/Users/helen/Downloads/wolfactiv_backend_complet/data/parfums_enrichi.csv",
        encoding="ISO-8859-1",
        sep=";"
    )

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


