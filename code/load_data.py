import pandas as pd
from pathlib import Path
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

df_meta = pd.read_csv(METADATA_PATH, usecols=colonnes_utiles)
all_texts = []

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

df_final = pd.merge(
    df_texts, 
    df_meta, 
    left_on="file_id", 
    right_on="id",
    how="inner"
)

print(f"Nombre de documents analysables: {len(df_final)}")