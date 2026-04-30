import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from paths import GRAPHS_FOLDER 
from code.load_data import df_final

# python -m code.descriptive_statistics

# Configuration esthétique
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)

# Calcul préalable : Création de la colonne du nombre de mots
df_final['word_count'] = df_final['text'].apply(lambda x: len(str(x).split()))

print("Génération des statistiques descriptives (3 graphiques)...")

# ==========================================
# Graphique 1 : Évolution temporelle
# ==========================================
plt.figure(figsize=(10, 5))
sns.countplot(data=df_final, x='date', color='steelblue')
plt.title("1. Volume de professions de foi par année électorale")
plt.xlabel("Année")
plt.ylabel("Nombre de documents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "1_evolution_temporelle.png", dpi=300)
plt.close()

# ==========================================
# Graphique 2 : Le paysage politique
# ==========================================
plt.figure(figsize=(10, 8))
top_partis = df_final['titulaire-soutien'].value_counts().nlargest(15).index
sns.countplot(
    data=df_final[df_final['titulaire-soutien'].isin(top_partis)], 
    y='titulaire-soutien', 
    order=top_partis, 
    palette="viridis", 
    hue='titulaire-soutien', 
    legend=False
)
plt.title("2. Les 15 partis politiques les plus représentés")
plt.xlabel("Nombre de candidats")
plt.ylabel("Parti Politique")
plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "2_repartition_partis.png", dpi=300)
plt.close()

# ==========================================
# Graphique 3 : Distribution de la longueur des textes (Ancien n°4)
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(data=df_final, x='word_count', bins=50, kde=True, color='purple')
plt.title("3. Distribution de la longueur des professions de foi")
plt.xlabel("Nombre de mots")
plt.ylabel("Fréquence")
plt.xlim(0, df_final['word_count'].quantile(0.99)) 
plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "3_distribution_mots.png", dpi=300)
plt.close()

# ==========================================
# Graphique 4 : Composition politique par année (Barres empilées)
# ==========================================
plt.figure(figsize=(12, 6))
TOP_X = 8
top_partis_stack = df_final['titulaire-soutien'].value_counts().nlargest(TOP_X).index
df_stack = df_final.copy()
df_stack['Parti_Regroupe'] = df_stack['titulaire-soutien'].apply(
    lambda x: x if x in top_partis_stack else 'Autres'
)
cross_tab = pd.crosstab(df_stack['date'], df_stack['Parti_Regroupe'])
ordre_colonnes = list(top_partis_stack) + ['Autres']
cross_tab = cross_tab.reindex(columns=[c for c in ordre_colonnes if c in cross_tab.columns])
cross_tab.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10', width=0.8, edgecolor='white', linewidth=0.5)

plt.title("4. Évolution du volume de textes et composition partisane")
plt.xlabel("Année électorale")
plt.ylabel("Nombre total de documents")
plt.xticks(rotation=45)
plt.legend(title="Parti Politique", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "4_evolution_composition_partis.png", dpi=300)
plt.close()
