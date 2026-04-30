from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from code.load_data import df_final
from paths import GRAPHS_FOLDER, STOPWORDS_PATH
from code.helpers import get_stopwords

# python -m code.main

stopwords = get_stopwords()

print("1. Vectorisation en cours (CountVectorizer)...")
vectorizer = CountVectorizer(
    max_df=0.3, 
    min_df=10, 
    stop_words=stopwords,
    token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{2,}\b"
)
dtm = vectorizer.fit_transform(df_final['text'])

print("2. Extraction des thèmes (LDA)...")
lda = LatentDirichletAllocation(
    n_components=14, 
    random_state=0 
)

topic_values = lda.fit_transform(dtm)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Thème {topic_idx + 1}: {', '.join(top_words)}")

print("3. Projection UMAP en cours (cela peut prendre 1 à 2 minutes)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=0)
embedding = reducer.fit_transform(topic_values)

df_final['x'] = embedding[:, 0]
df_final['y'] = embedding[:, 1]
df_final['theme_dominant'] = topic_values.argmax(axis=1)

print("4. Génération de la carte sémantique et de la distribution des thèmes...")

theme_labels = {}
feature_names = vectorizer.get_feature_names_out() 

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-4:-1] 
    top_words = [feature_names[i] for i in top_words_idx]
    theme_labels[topic_idx] = f"Thème {topic_idx + 1} : {', '.join(top_words)}"

df_map = df_final.copy()

mots_a_exclure = ['non mentionné', 'sans étiquette', 'indépendant', 'inconnu', 'non renseigné']
df_map = df_map[~df_map['titulaire-soutien'].astype(str).str.lower().isin(mots_a_exclure)]

TOP_X_PARTIS = 5
comptage = df_map['titulaire-soutien'].value_counts()
partis_principaux = comptage.head(TOP_X_PARTIS).index
df_map = df_map[df_map['titulaire-soutien'].isin(partis_principaux)]

df_map['Nom_Theme'] = df_map['theme_dominant'].map(theme_labels)

GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)

sns.set_style("ticks") 
plt.figure(figsize=(15, 10)) 

df_plot = df_map.sample(frac=0.3, random_state=42)

ax = sns.scatterplot(
    data=df_plot, 
    x='x', 
    y='y', 
    hue='Nom_Theme',          
    style='titulaire-soutien', 
    palette='tab20', 
    s=75,                      
    alpha=0.8,                 
    edgecolor='white',         
    linewidth=0.5
)

plt.title(f"Paysage Idéologique : Top {TOP_X_PARTIS} des Partis Politiques (Sampling 30%)", fontsize=16, pad=20, fontweight='bold')
plt.xlabel("") 
plt.ylabel("") 
plt.xticks([]) 
plt.yticks([]) 
sns.despine(left=True, bottom=True) 

handles, labels = ax.get_legend_handles_labels()

try:
    split_idx = labels.index('titulaire-soutien')
    
    leg_themes = ax.legend(
        handles[1:split_idx], labels[1:split_idx], 
        title="Domaines (LDA)", 
        bbox_to_anchor=(1.02, 1), loc='upper left', 
        frameon=False, 
        fontsize='small', 
        title_fontproperties={'weight': 'bold', 'size': 12}
    )
    ax.add_artist(leg_themes) 
    
    leg_partis = ax.legend(
        handles[split_idx+1:], labels[split_idx+1:], 
        title="Partis Politiques", 
        bbox_to_anchor=(1.02, 0.4), loc='upper left', 
        frameon=False, 
        markerscale=1.5, 
        title_fontproperties={'weight': 'bold', 'size': 12}
    )
    
except ValueError:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "semantic_map_sampled.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(14, 8))
order_themes = [theme_labels[i] for i in range(len(theme_labels))]

sns.countplot(
    data=df_map, 
    y='Nom_Theme', 
    hue='titulaire-soutien', 
    order=order_themes,
    palette='tab10'
)

tableau_croise = pd.crosstab(df_map['Nom_Theme'], df_map['titulaire-soutien'])

order_themes = [theme_labels[i] for i in range(len(theme_labels))]
tableau_croise = tableau_croise.reindex(order_themes)

ax = tableau_croise.plot(
    kind='barh', 
    stacked=True, 
    figsize=(14, 10), 
    colormap='tab10', 
    width=0.8, 
    edgecolor='white',
    linewidth=0.5
)

plt.title("Distribution des Thèmes par Parti Politique", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Nombre total de professions de foi", fontsize=12)
plt.ylabel("")

plt.gca().invert_yaxis()

plt.legend(title="Partis Politiques", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
sns.despine()

plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "theme_distribution_stacked.png", dpi=300, bbox_inches='tight')

print("5. Génération de la carte sémantique par parti (Small Multiples)...")

unique_themes = df_map['Nom_Theme'].unique()
theme_palette = dict(zip(unique_themes, sns.color_palette("tab20", len(unique_themes))))

fig = plt.figure(figsize=(20, 12))
fig.suptitle("Paysage Idéologique : Mise en évidence par Parti Politique", fontsize=20, fontweight='bold', y=1.02)

for i, parti in enumerate(partis_principaux):
    ax = plt.subplot(2, 3, i + 1)
    
    sns.scatterplot(
        data=df_plot, x='x', y='y', 
        color='lightgray', s=30, alpha=0.3, edgecolor='none', ax=ax, legend=False
    )
    
    df_parti = df_plot[df_plot['titulaire-soutien'] == parti]
    sns.scatterplot(
        data=df_parti, x='x', y='y', 
        hue='Nom_Theme', palette=theme_palette, 
        s=60, alpha=0.9, edgecolor='white', linewidth=0.5, ax=ax, legend=False
    )
    
    ax.set_title(parti.upper(), fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True, ax=ax)

if len(partis_principaux) == 5:
    plt.subplot(2, 3, 6).axis('off')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in theme_palette.values()]
    plt.legend(handles, theme_palette.keys(), title="Domaines (LDA)", loc='center', frameon=False, fontsize=10, title_fontproperties={'weight':'bold', 'size':12})

plt.tight_layout()
plt.savefig(GRAPHS_FOLDER / "semantic_map_small_multiples.png", dpi=300, bbox_inches='tight')
print("Terminé ! Graphiques sauvegardés dans le dossier figures.")