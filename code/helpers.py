import nltk
from nltk.corpus import stopwords
from paths import STOPWORDS_PATH

def get_stopwords():
    nltk.download('stopwords', quiet=True)
    all_stopwords = stopwords.words('french') + stopwords.words('german')
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            mots_politiques = [ligne.strip() for ligne in f if ligne.strip()] 
            all_stopwords.extend(mots_politiques)
    except FileNotFoundError:
        pass
    return list(set(all_stopwords))