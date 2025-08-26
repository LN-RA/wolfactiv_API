Wolfactiv MBTI API
API FastAPI qui :
infère un type MBTI depuis des réponses texte,


transforme ce type en vecteur olfactif,


projette ce vecteur via une matrice de similarité,


calcule les similarités avec une base de parfums,


renvoie un Top 5 de recommandations + persona + radar data
 et enregistre les résultats dans Supabase.


🗂 Structure
wolfactiv_mbti_api/
├─ main.py                # API FastAPI + logique MBTI + accès Supabase
├─ recommender.py         # projection (S @ u) + similarités cosinus
├─ requirements.txt       # dépendances
├─ .env                   # clés et chemins (non commité)
└─ __pycache__/           # fichiers Python compilés

🔧 Prérequis
Python 3.10+


Un compte OpenAI (clé API)


Un projet Supabase (URL + clé service/anon selon ton usage)


Deux fichiers de données locaux (encodage ISO-8859-1, séparateur ;) :


similarité matrice.csv (matrice S, shape: n_parfums × n_notes)


parfums_enrichi.csv (colonnes notes + Marque, Nom du Parfum, images parfums/Image, Lien de redirection/URL)


⚠️ Aujourd’hui les chemins sont en dur dans recommender.py. Voir section Configuration pour les passer en variables d’environnement.
📦 Installation
Windows (PowerShell)
cd .\wolfactiv_mbti_api\
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

macOS / Linux
cd wolfactiv_mbti_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

🔐 Configuration (.env)
Crée un fichier .env à la racine avec :
# API Keys
OPENAI_API_KEY=sk-...

# Supabase
SUPABASE_KEY=eyJhbGci...   # ta clé (service ou anon selon ce que tu fais)
SUPABASE_URL=https://oimzzeyjjovxdhuscmqw.supabase.co

# Données locales (remplace par tes chemins)
SIM_MATRIX_CSV=C:/Users/helen/Downloads/wolfactiv_backend_complet/data/similarité matrice.csv
PARFUMS_CSV=C:/Users/helen/Downloads/wolfactiv_backend_complet/data/parfums_enrichi.csv

# Encodage et séparateur (optionnel si tu veux rendre ça paramétrable)
CSV_ENCODING=ISO-8859-1
CSV_SEP=;

Et modifie recommender.py pour lire ces variables plutôt que des chemins en dur (voir Améliorations conseillées).
🚀 Lancement
uvicorn main:app --reload --port 8000

API disponible sur: http://127.0.0.1:8000
CORS
CORS est ouvert à * dans main.py. Mets ton domaine front en prod :
allow_origins=["https://ton-domaine.app"]

📡 Endpoint
POST /submit_quiz
Body:


{
  "email": "user@example.com"
}

Logique:


Récupère le dernier enregistrement quiz_results dans Supabase pour cet email


attend les champs :
 personality_answers (string|list),
 disliked_odors (array),
 happy_memory_odor (string),
 strong_memory (bool),
 strong_memory_odor (string).


Appelle OpenAI pour inférer le MBTI depuis personality_answers.


Charge encoding_perso.xlsx pour convertir le MBTI ➜ vecteur.


Ajuste le vecteur selon les goûts/dégoûts.


Projette via la matrice S (CSV).


Calcule similarité cosinus avec parfums_enrichi.csv et renvoie Top 5.


Écrit dans Supabase :


table resultats_mbti : { email, mbti_result, vector }


table results : { mbti, character_name, quote, perfumes, radar_data }


Réponse (ex) :


{
  "email": "user@example.com",
  "mbti": "ENFP",
  "character_name": "Le Visionnaire",
  "quote": "La vie est une aventure audacieuse ou rien du tout.",
  "vector": { "...": 0.5 },
  "radar_chart": { "Florale": 72.1, "...": 33.5 },
  "top_perfumes": [
    { "name": "Marque - Nom", "score": 92.35, "image_url": "...", "link": "..." }
  ]
}

🧠 Détails techniques
recommender.py
get_u_final(u_vector)


charge S depuis SIM_MATRIX_CSV


vérifie la compatibilité des dimensions


retourne S @ u


calculate_similarities(u_final)


charge parfums_enrichi.csv


normalise/renomme colonnes :


ï»¿images parfums → Image, Lien de redirection → URL


mappe les familles olfactives avec difflib.get_close_matches


calcule cosine similarity ligne par ligne contre u_final


renvoie Top 5 (score en %)


main.py
OpenAI: gpt-4 via chat.completions.create pour inférer le MBTI


Supabase: lecture quiz_results, écriture dans resultats_mbti et results


MBTI ➜ vecteur: lit encoding_perso.xlsx (colonne MBTI + colonnes notes)


🧪 Test rapide
PowerShell (exemple avec Invoke-RestMethod)
$body = @{ email = "user@example.com" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/submit_quiz `
  -ContentType "application/json" `
  -Body $body

curl
curl -X POST http://127.0.0.1:8000/submit_quiz \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com"}'

🗄 Schéma Supabase attendu
quiz_results (lecture)


email (text)


submitted_at (timestamp)


personality_answers (text|json)


disliked_odors (json/array)


happy_memory_odor (text)


strong_memory (bool)


strong_memory_odor (text)


resultats_mbti (écriture)


email (text)


mbti_result (text)


vector (json/array)


results (écriture)


mbti (text)


character_name (text)


quote (text)


perfumes (json)


radar_data (json)


✅ Améliorations conseillées (quick wins)
Chemins de données via .env (voir variables plus haut)
 Dans recommender.py :

 import os
SIM_PATH = os.getenv("SIM_MATRIX_CSV")
PARFUMS_PATH = os.getenv("PARFUMS_CSV")
ENCODING = os.getenv("CSV_ENCODING", "ISO-8859-1")
SEP = os.getenv("CSV_SEP", ";")


Modèle OpenAI : migrer vers gpt-4o / gpt-4.1 si dispo, ou limiter tokens.


Validation Pydantic pour le payload Supabase.


Sécurité : restreindre allow_origins et ne jamais commiter .env.


Logs : remplacer print par logging.


Tests : ajouter des tests unitaires pour cosine_similarity, mapping de colonnes, etc.


🐛 Dépannage
ValueError: Incompatibilité dimensions
 ➜ La taille de u_vector ne correspond pas au nombre de colonnes notes sélectionnées. Vérifier l’ordre et le mapping des familles.


Colonnes non trouvées (parfums_enrichi.csv)
 ➜ Vérifier l’encodage/sep et les renommages (Image, URL).


OpenAI Unauthorized
 ➜ Vérifie OPENAI_API_KEY dans .env.


Supabase
 ➜ La clé et les noms de tables doivent être corrects, et la politique RLS adaptée.
