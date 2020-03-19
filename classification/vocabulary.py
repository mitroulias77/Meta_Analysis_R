'''Δημιουργία Λεξικού'''
from functools import reduce
import el_core_news_sm
import operator
from collections import Counter
nlp = el_core_news_sm.load()
def tokenizeSentences(sent):
    doc = nlp(sent)
    sentences = [sent.string.strip() for sent in doc]
    return sentences

Xs = []
for texts in dataframe.Concultatory:
    Xs.append(tokenizeSentences(texts))
vocab = sorted(reduce(lambda x, y: x | y, (set(words) for words in Xs)))
len(vocab)


def word_freq(Xs, num):
    all_words = [words.lower() for sentences in Xs for words in sentences]
    sorted_vocab = sorted(dict(Counter(all_words)).items(), key=operator.itemgetter(1))
    final_vocab = [k for k,v in sorted_vocab if v > num]
    word_idx = dict((c, i + 1) for i, c in enumerate(final_vocab))
    return final_vocab, word_idx

final_vocab, word_idx = word_freq(Xs,2)
vocab_len = len(final_vocab)