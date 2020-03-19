'''Python script δημιουργίας word vectors με Word2Vec'''
import  numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import gensim
from gensim.models import KeyedVectors
warnings.filterwarnings(action='ignore')

sample = open("data/decisions.txt", "r", encoding="utf8")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Δημιουργία CBOW model
model = gensim.models.Word2Vec(data, min_count = 1,
                              size = 100, window = 5)

WordVectorz=dict(zip(model.wv.index2word,model.wv.vectors))
model.save("data/embeddings/CBOW.bin")

vector_dim=100
model = KeyedVectors.load('data/embeddings/CBOW.bin')
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))

for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


transformer = FunctionTransformer(embedding_matrix)

combined_df=decisions_new['Concultatory'].append(decisions_new['Title'])

class AverageEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim =100 # because we use 100 embedding points

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

#########################################################

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                             window = 5, sg = 1)
filename = "data/embeddings/skipgram.txt"
model2.save_word2vec_format(filename, binary=False)
model2.save("data/embeddings/skipgram.model")
model2.save("data/embeddings/skipgram.bin")
similar_words = model2.most_similar('υπαλληλος')
print(similar_words)

