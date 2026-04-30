import pandas as pd
from paths import METADATA_PATH, DATA_FOLDER

# python -m code.load_data

colonnes_utiles = [
    'id', 
    'date', 
    'departement-nom', 
    'titulaire-soutien', 
    'titulaire-profession',
    'titulaire-sexe'
]

print("Chargement des métadonnées...")
df_meta = pd.read_csv(METADATA_PATH, usecols=colonnes_utiles, low_memory=False)
all_texts = []

print("Chargement des textes...")
for file_path in DATA_FOLDER.glob("**/*.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        all_texts.append({
            "file_id": file_path.stem, 
            "text": content
        })
    except Exception as e:
        pass

df_texts = pd.DataFrame(all_texts)

# Fusion (Inner join pour ne garder que les textes qui ont bien une ligne de métadonnée)
df_final = pd.merge(
    df_texts, 
    df_meta, 
    left_on="file_id", 
    right_on="id",
    how="inner"
)

nb_final = len(df_final)

# --- Affichage clair du résumé ---
print("\n=== RÉSUMÉ DU CHARGEMENT DES DONNÉES ===")
print(f"Textes trouvés dans les dossiers : {len(df_texts)}")
print(f"Métadonnées correspondantes      : {len(df_meta)}")
print("----------------------------------------")
print(f"Nombre total de documents conservés pour l'entraînement LDA : {nb_final}")
print("========================================\n")