from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from pathlib import Path
from openai import OpenAI
import pandas as pd
from .recommender import get_u_final, calculate_similarities

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par ton domaine frontend si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "wolfactiv_mbti_api"}

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


# Charger les variables d'environnement depuis .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Clés API
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = "https://oimzzeyjjovxdhuscmqw.supabase.co"
supabase: Client = create_client(supabase_url, supabase_key)

class QuizRequest(BaseModel):
    email: str

# Fonction d'inférence MBTI
def infer_mbti_from_answers(answers_text: str) -> str:
    prompt = f'''
Tu es un expert en psychologie MBTI.
Voici des réponses à des questions de personnalité :
"{answers_text}"

Analyse-les et indique uniquement le type MBTI (parmi INFP, ESTJ, ENTP...) sans explication.
''' 
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un expert MBTI."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip().upper()

# Récupération du vecteur MBTI
def get_vector_from_mbti(mbti: str) -> dict:
    try:
        excel_path = r"C:\Users\helen\OneDrive\Documents\Entreprise\wolfactiv\wolfactool\encoding_perso.xlsx"
        df = pd.read_excel(excel_path)
        if "MBTI" not in df.columns:
            raise ValueError("Colonne 'MBTI' absente dans le fichier.")
        row = df[df["MBTI"].str.upper() == mbti.upper()]
        if row.empty:
            raise ValueError(f"Type MBTI {mbti} non trouvé.")
        vector = row.iloc[0, 1:].astype(float).to_dict()
        return vector
    except Exception as e:
        raise RuntimeError(f"Erreur dans le chargement du vecteur MBTI : {e}")

# Ajustement du vecteur avec les goûts utilisateur
def adjust_vector(vector: dict, disliked: list, happy: str, strong: bool, strong_odor: str) -> dict:
    for note in disliked:
        if note in vector:
            vector[note] = 0.0
    if happy in vector:
        vector[happy] += 1.0
    if strong and strong_odor in vector:
        vector[strong_odor] += 1.0
    return vector

# Persona & citation associée au MBTI
def get_persona_and_quote(mbti: str):
    mbti_dict = {
        "INFP": ("Le Poète", "Ce que l'on voit n’est qu’une illusion, ce que l’on ressent est vérité."),
        "ENTP": ("L’Explorateur", "Je ne cherche pas, je trouve."),
        "ESTJ": ("Le Stratège", "L’ordre n’est pas une contrainte, c’est une force."),
        "ENFP": ("Le Visionnaire", "La vie est une aventure audacieuse ou rien du tout."),
        "ISFJ": ("Le Gardien", "Petits gestes, grands cœurs."),
        "INTJ": ("L'Architecte", "Tout ce qui mérite d’être fait mérite d’être bien fait."),
        # ... ajoute d'autres profils selon ta base
    }
    return mbti_dict.get(mbti, ("Inconnu", "Aucune citation disponible."))

# Endpoint principal
from fastapi.encoders import jsonable_encoder

@app.post("/submit_quiz")  # ✅
def submit_quiz(data: QuizRequest):
    try:
        print("🔍 Recherche quiz pour :", data.email)

        result = supabase.table("quiz_results")\
            .select("*")\
            .eq("email", data.email)\
            .order("submitted_at", desc=True)\
            .limit(1)\
            .execute()

        if not result.data:
            return {"error": "Aucune donnée pour cet email"}

        user_data = result.data[0]
        answers = user_data.get("personality_answers", "")
        if isinstance(answers, list):
            answers = "\n".join(answers)

        if not answers:
            return {"error": "Réponses manquantes"}

        # Étape 1 : Inférence MBTI
        mbti = infer_mbti_from_answers(answers)
        print("🧬 MBTI :", mbti)

        # Étape 2 : Données sensorielles
        disliked = user_data.get("disliked_odors", [])
        happy = user_data.get("happy_memory_odor", "")
        strong = user_data.get("strong_memory", False)
        strong_odor = user_data.get("strong_memory_odor", "")

        # Étape 3 : Vecteur brut et ajusté
        vector = get_vector_from_mbti(mbti)
        vector = adjust_vector(vector, disliked, happy, strong, strong_odor)
        print("📊 Vecteur ajusté :", vector)

        # Étape 4 : Projection (matrice S)
        vector_np = list(vector.values())
        u_final = get_u_final(vector_np)
        print("🎯 Vecteur projeté :", u_final)

        # Étape 5 : Matching parfums
        top_perfumes = calculate_similarities(u_final)
        print("💡 Parfums bruts retournés par calculate_similarities :")
        for i, p in enumerate(top_perfumes):
            print(f"{i+1}: {p}")

        # Étape 6 : Sauvegarde MBTI brut
        supabase.table("resultats_mbti").insert({
            "email": data.email,
            "mbti_result": mbti,
            "vector": vector_np
        }).execute()

        # Étape 7 : Persona + Radar Chart
        character_name, quote = get_persona_and_quote(mbti)
# Suite de analyze_mbti
        radar_data = {k: round(v * 100, 2) for k, v in zip(vector.keys(), u_final)}

        # Étape 8 : JSON des parfums
        perfume_json = [
            {
                "name": p["parfum"],
                "score": p["similarité"],
                "image_url": p.get("image", ""),
                "link": p.get("url", "")
            }
            for p in top_perfumes
        ]

        # Étape 9 : Sauvegarde enrichie
        supabase.table("results").insert({
            "mbti": mbti,
            "character_name": character_name,
            "quote": quote,
            "perfumes": perfume_json,
            "radar_data": radar_data
        }).execute()

        # Étape 10 : Réponse JSON
        return jsonable_encoder({
            "email": data.email,
            "mbti": mbti,
            "character_name": character_name,
            "quote": quote,
            "vector": vector,
            "radar_chart": radar_data,
            "top_perfumes": perfume_json
        })

    except Exception as e:
        print("❌ Erreur dans analyze_mbti:", e)
        return {
            "error": "Erreur interne",
            "details": str(e)
        }


