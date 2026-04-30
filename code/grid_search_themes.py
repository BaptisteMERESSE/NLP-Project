import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from code.load_data import df_final
from paths import GRAPHS_FOLDER
from code.helpers import get_stopwords

# python -m code.grid_search_themes

def main():
    stopwords = get_stopwords()

    vectorizer = CountVectorizer(
        max_df=0.4, 
        min_df=5, 
        stop_words=stopwords,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{2,}\b"
    )
    dtm = vectorizer.fit_transform(df_final['text'])
    feature_names = vectorizer.get_feature_names_out()

    analyzer = vectorizer.build_analyzer()
    texts = [analyzer(text) for text in df_final['text']]
    dictionary = Dictionary(texts)

    topic_range = range(4, 25, 2)
    coherence_scores = []

    for n_topics in topic_range:
        print(f"Évaluation pour {n_topics} thèmes...")
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(dtm)
        
        topic_words = []
        for topic in lda.components_:
            top_words_idx = topic.argsort()[:-21:-1]
            topic_words.append([feature_names[i] for i in top_words_idx])
            
        cm = CoherenceModel(
            topics=topic_words, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v',
            processes=1
        )
        coherence_scores.append(cm.get_coherence())

    sns.set_style("ticks")
    plt.figure(figsize=(10, 6))

    plt.plot(topic_range, coherence_scores, marker='o', linewidth=2, color='#1f77b4')

    plt.title("Évolution du score de cohérence (C_v) selon le nombre de thèmes", fontsize=14, fontweight='bold')
    plt.xlabel("Nombre de thèmes", fontsize=12)
    plt.ylabel("Score de Cohérence", fontsize=12)
    plt.xticks(list(topic_range))
    sns.despine()

    plt.tight_layout()
    GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAPHS_FOLDER / "coherence_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()