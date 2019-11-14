import keras.optimizers
import warnings
from keras import Sequential, Model, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from functools import reduce
from keras_preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")
import el_core_news_sm
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

from Reports.helper_functions import *
overall_f1_score_v1_cv = make_scorer(overall_f1_score_v1, greater_is_better=True)

my_data_train = pd.read_csv('data/preprocessed/decisions_lemmas_train_preprocessed.csv')
my_data_test = pd.read_csv('data/preprocessed/decisions_lemmas_test_preprocessed.csv')

train_X, train_y = my_data_train['Concultatory'], my_data_train.drop(['Concultatory','Title'], axis=1)
test_X, test_y = my_data_test['Concultatory'], my_data_test.drop(['Concultatory','Title'], axis=1)

nlp = el_core_news_sm.load()

def tokenizeSentences(sent):
    doc = nlp(sent)
    sentences = [sent.string.strip() for sent in doc]
    return sentences


'''
Συνάρτηση για να πάρουμε λέξεις που τουλάχιστον εμφανίστηκαν περισσότερο από μία φορά στο vocab μας. 
Αντί να επιλέξουμε τις  500 πιο συχνές λέξεις. θα υπάρξουν πολλές λέξεις 
που θα εξαλειφθούν αυθαίρετα μόλις φτάσουμε το όριο των 500 λέξεων. 
(Πολλές λέξεις έχουν πολύ παρόμοιες μετρήσεις!)
'''
import operator
from collections import Counter

def word_freq(Xs, num):
    all_words = [words.lower() for sentences in Xs for words in sentences]
    sorted_vocab = sorted(dict(Counter(all_words)).items(), key=operator.itemgetter(1))
    final_vocab = [k for k,v in sorted_vocab if v > num]
    word_idx = dict((c, i + 1) for i, c in enumerate(final_vocab))
    return final_vocab, word_idx

'''
H παρακάτω συνάρτηση θα μοιράσει τις λέξεις που έχουμε 
'''
def vectorize_sentences(data, word_idx, final_vocab, maxlen=50):
    X = []
    paddingIdx = len(final_vocab)+2
    for sentences in data:
        x=[]
        for word in sentences:
            if word in final_vocab:
                x.append(word_idx[word])
            elif word.lower() in final_vocab:
                x.append(word_idx[word.lower()])
            else:
                x.append(paddingIdx)
        X.append(x)
    return (pad_sequences(X, maxlen=maxlen))

Xs_train = []
for texts in train_X:
    Xs_train.append(tokenizeSentences(texts))
vocab = sorted(reduce(lambda x, y: x | y, (set(words) for words in Xs_train)))
len(vocab)

final_vocab, word_idx = word_freq(Xs_train,2)
vocab_len = len(final_vocab)

Xs_test = []
for texts in test_X:
    Xs_test.append(tokenizeSentences(texts))
vocab2 = sorted(reduce(lambda x, y: x | y, (set(words) for words in Xs_test)))
len(vocab2)

final_vocab2, word_idx2 = word_freq(Xs_test,2)
vocab_len2 = len(final_vocab2)

# new_y = pd.DataFrame(decisions_new.drop(['Title', 'Concultatory'], axis=1, inplace=True))
X_train_vectorized = vectorize_sentences(Xs_train, word_idx, final_vocab)
X_test_vectorized = vectorize_sentences(Xs_test, word_idx2, final_vocab2)

lemma_columns = train_y.columns

train_y_labels= train_y.groupby(list(lemma_columns)).ngroup()
y_labels_lut = train_y.copy(deep=True)
y_labels_lut['Labels'] = train_y_labels
y_labels_lut = y_labels_lut.drop_duplicates()
y_labels_lut = y_labels_lut.reset_index(drop=True).set_index('Labels').sort_index()

from keras.utils import np_utils
num_classes = y_labels_lut.shape[0]
train_y_onehot = np_utils.to_categorical(train_y_labels, num_classes=num_classes)

def gen_model(optimizer):
  model = Sequential()
  model.add(Dense(1024, activation='relu', input_shape=(50,)))
  model.add(Dropout(0.2))
  model.add(Dense(22, activation='softmax'))
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

lr_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

epochs, batch_size = 100, 32
model = gen_model(keras.optimizers.SGD(lr=1))
model.fit(X_train_vectorized, train_y_onehot,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.3,
          callbacks=[lr_reduction])




y_pred = model.predict(X_test_vectorized)
y_pred_label = pd.DataFrame(np.argmax(y_pred, axis=1))
predictions = pd.merge(y_pred_label, y_labels_lut, how='left', left_on=0, right_on='Labels')[lemma_columns]
accuracy(test_y, predictions)


from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import concatenate
n_in = 50
EMBEDDING_DIM=100

filter_sizes = (2,4,5,8)
dropout_prob = [0.4,0.5]

graph_in = Input(shape=(n_in, EMBEDDING_DIM))
convs = []
avgs = []

from keras.layers import concatenate
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=32,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=n_in-fsz+1)(conv)
    flattenMax = Flatten()(pool)
    convs.append(flattenMax)

if len(filter_sizes) > 1:
    out = concatenate(convs, axis=1)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out, name="graphModel")
model = Sequential()
model.add(Embedding(input_dim=vocab_len+3, #size of vocabulary
                 output_dim = EMBEDDING_DIM,
                input_length = n_in,
                 trainable=True))
model.add(Dropout(dropout_prob[0]))
model.add(graph)
model.add(Dense(256))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(train_y_onehot.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='data/models/model-cnn_multi_filter.h5', save_best_only=True)
]
history = model.fit(X_train_vectorized, train_y_onehot,
                    class_weight='auto',
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks)

y_pred = model.predict(X_test_vectorized)
y_pred_label = pd.DataFrame(np.argmax(y_pred, axis=1))
predictions = pd.merge(y_pred_label, y_labels_lut, how='left', left_on=0, right_on='Labels')[lemma_columns]
accuracy(test_y, predictions)