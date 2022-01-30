import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
import numpy as np
import nltk
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

#To improve the value of the silhouette is very low, though, the exercise is kind of complete
nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            lemmas.append(lemma)
        return lemmas


path = "newsgroups/T-newsgroups/"
dirs = os.listdir(path)


df = pd.DataFrame()


for el in dirs:
    with open(os.path.join(path, el), 'r') as fp:
        df.loc[el, 'text'] = fp.read()


corpus = np.array(df.loc[:, 'text'])

lemmaTokenizer = LemmaTokenizer()
vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'))

tfidf_x = vectorizer.fit_transform(corpus)

labels = DBSCAN(min_samples=20).fit_predict(tfidf_x)
sil_score = silhouette_score(tfidf_x, labels)

print(sil_score)
print(vectorizer.get_feature_names())

