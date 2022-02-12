import multiprocessing as mp
import re
import nltk
from gensim.models import Word2Vec
import pandas as pd
import random



data = pd.read_csv('C:/Users/Win 10/Desktop/lyrics-data.csv')


data = data[['SName','Lyric']]


data = data.dropna()

data = data.drop_duplicates()


data = data.reset_index(drop=True)

data['text'] = data['SName'] + " " + data['Lyric']


text = str(data['text'])



def clean(text):
    text = text.lower()
    text = re.sub("[0-9]", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

text = clean(text)    
sentences = nltk.sent_tokenize(text)
words = [nltk.word_tokenize(word) for word in sentences]



word2vec = Word2Vec(words, min_count=2)

def apk(actual, predicted, k=10):
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

import random

class Document:
    def __init__(self, title, text):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']

my_index=[]
def build_index():
    for i in range(df.shape[0]):
        index_.append(Document(df['SName'][i], df['Lyric'][i]))
        
def score(query, document):
    score_1 = 1
    score_2 = 0
    for i in list(dict(word2vec.wv.most_similar(query,topn=50)).keys()):
        if i.lower() in " ".join(document.format(query)):
            score_1 += 1
            score_2 += dict(word2vec.wv.most_similar(query,topn=50)).get(i) 
    return score_2/score_1

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    candidates = []
    for doc in my_index:
        if query.lower() in doc.title.lower() or query in doc.text.lower():
            candidates.append(doc)
    return candidates[:5]


