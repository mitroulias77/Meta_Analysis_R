from functools import reduce
from keras import Input, Model
from keras.layers import GlobalMaxPool1D, Conv1D,LSTM, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from itertools import chain
import warnings
import pandas as pd
from Reports.helper_functions import accuracy
from keras.callbacks import ReduceLROnPlateau
import keras.optimizers
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
warnings.filterwarnings("ignore")
import numpy as np
import nltk
import el_core_news_sm

nlp = el_core_news_sm.load()

dataframe=pd.read_csv("data/preprocessed/decisions_lemmas.csv")

def tokenizeSentences(sent):
    doc = nlp(sent)
    sentences = [sent.string.strip() for sent in doc]
    return sentences
###########################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

maxlen = 180
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(dataframe.Concultatory)
def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)

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
###################################################
nsk_list = dataframe['Category'].tolist()
nsk_list = [(str(x)).split(',') for x in nsk_list]

for idx, lst in enumerate(nsk_list):
    cats = [x.strip() for x in lst]
    dataframe.at[idx, 'Category'] = cats
nsk_list = list(chain.from_iterable(nsk_list))
nsk_list = [x.strip() for x in nsk_list]
nsk_set = set(nsk_list)

categories = (list(nsk_set))

categories.sort()
categories_accumulator = [0]*len(categories)
for index, row in dataframe.iterrows():
    cats = row['Category']
    for cat in cats:
        accumulator_idx = categories.index(cat)
        categories_accumulator[accumulator_idx] +=1

sorted_indexes = sorted(range(len(categories_accumulator)), key=lambda k: categories_accumulator[k], reverse=True)
categories_accumulator.sort(reverse=True)
categories.sort()


lemmata = []
lemmata.extend(dataframe['Category'].tolist())
lemmata = [list(filter(None, empty)) for empty in lemmata]

dataframe['New_Category'] = lemmata
decisions_new = dataframe[~(dataframe['New_Category'].str.len() == 0)]

all_categories =  sum(lemmata,[])
len(all_categories)
len(set(all_categories))

decisions_new['Lemmata'] = decisions_new['New_Category'].apply(lambda text : len(str(text).split(',')))

decisions_new.Lemmata.value_counts()
all_categories_new =  nltk.FreqDist(all_categories)

count_lemmas = pd.DataFrame({'Lemma': list(all_categories_new.keys()),
                                 'Count': list(all_categories_new.values())})
count_lemmas_sorted = count_lemmas.sort_values(['Count'],ascending=False)
lemma_counts = count_lemmas_sorted['Count'].values

sorted_idx = count_lemmas_sorted.index
sorted_idx = list(sorted_idx)

lemmata_new = list(count_lemmas['Lemma'])
new_lemmata = [lemmata_new[word] for word in sorted_idx[:20]]
new_categories = decisions_new['New_Category']

for idx, row in decisions_new.iterrows():
    cats = row['New_Category']
    new_cats = []
    for cat in cats:
        if cat in new_lemmata:
            new_cats.append(cat)
    decisions_new.at[idx, 'New_Category'] = new_cats

columns = ['index', 'Category', 'Lemmata']
decisions_new.drop(columns, axis=1, inplace=True)

decisions_new = decisions_new[decisions_new['New_Category'].map(lambda d: len(d))> 0]
decisions_new = decisions_new.reset_index(drop=True)

y = np.array([np.array(x) for x in decisions_new.New_Category.values.tolist()])
mlb = MultiLabelBinarizer()
y_1= mlb.fit_transform(y)
lemma_columns = mlb.classes_

decisions_new = decisions_new.join(pd.DataFrame(mlb.fit_transform(decisions_new.pop('New_Category')),
                                                columns=mlb.classes_,
                                                index=decisions_new.index))
Xs = []
for texts in decisions_new.Concultatory:
    Xs.append(tokenizeSentences(texts))
vocab = sorted(reduce(lambda x, y: x | y, (set(words) for words in Xs)))
len(vocab)

final_vocab, word_idx = word_freq(Xs,2)
vocab_len = len(final_vocab)

# new_y = pd.DataFrame(decisions_new.drop(['Title', 'Concultatory'], axis=1, inplace=True))
train_data = vectorize_sentences(Xs, word_idx, final_vocab)


x_train, x_test, y_train, y_test = train_test_split(train_data, y_1, test_size=0.1, random_state=42)

y_train_df = pd.DataFrame(y_train)
y_train_df.columns = ['ΑΚΙΝΗΤΑ', 'ΑΝΑΚΛΗΣΗ', 'ΑΠΑΛΛΑΓΕΣ ΜΕΙΩΣΕΙΣ ΕΚΠΤΩΣΕΙΣ',
       'ΑΠΟΔΟΧΕΣ ΕΠΙΔΟΜΑΤΑ', 'ΑΠΟΖΗΜΙΩΣΗ', 'ΑΡΜΟΔΙΟΤΗΤΑ', 'ΔΗΜΟΣΙΑ ΕΡΓΑ',
       'ΔΗΜΟΣΙΟ', 'ΔΙΑΓΩΝΙΣΜΟΣ ΜΕΙΟΔΟΣΙΑΣ ΠΛΕΙΟΔΟΣΙΑΣ',
       'ΔΙΟΡΙΣΜΟΣ ΠΡΟΣΛΗΨΗ', 'ΕΝΟΠΛΕΣ ΔΥΝΑΜΕΙΣ', 'ΕΤΑΙΡΕΙΕΣ ΑΝΩΝΥΜΕΣ',
       'ΚΡΑΤΙΚΕΣ ΠΡΟΜΗΘΕΙΕΣ', 'ΜΕΤΑΒΙΒΑΣΗ ΕΚΠΟΙΗΣΗ',
       'ΜΕΤΑΤΑΞΗ ΚΑΤΑΤΑΞΗ ΕΝΤΑΞΗ', 'ΠΡΟΘΕΣΜΙΑ', 'ΠΡΟΣΩΠΙΚΟ ΥΠΑΛΛΗΛΟΙ',
       'ΣΥΜΒΑΣΕΙΣ', 'ΣΥΜΜΟΡΦΩΣΗ ΔΙΟΙΚΗΣΕΩΣ', 'ΥΠΑΛΛΗΛΟΙ ΔΗΜΟΣΙΟΙ']
y_test_df = pd.DataFrame(y_test)
y_test_df.columns = ['ΑΚΙΝΗΤΑ', 'ΑΝΑΚΛΗΣΗ', 'ΑΠΑΛΛΑΓΕΣ ΜΕΙΩΣΕΙΣ ΕΚΠΤΩΣΕΙΣ',
       'ΑΠΟΔΟΧΕΣ ΕΠΙΔΟΜΑΤΑ', 'ΑΠΟΖΗΜΙΩΣΗ', 'ΑΡΜΟΔΙΟΤΗΤΑ', 'ΔΗΜΟΣΙΑ ΕΡΓΑ',
       'ΔΗΜΟΣΙΟ', 'ΔΙΑΓΩΝΙΣΜΟΣ ΜΕΙΟΔΟΣΙΑΣ ΠΛΕΙΟΔΟΣΙΑΣ',
       'ΔΙΟΡΙΣΜΟΣ ΠΡΟΣΛΗΨΗ', 'ΕΝΟΠΛΕΣ ΔΥΝΑΜΕΙΣ', 'ΕΤΑΙΡΕΙΕΣ ΑΝΩΝΥΜΕΣ',
       'ΚΡΑΤΙΚΕΣ ΠΡΟΜΗΘΕΙΕΣ', 'ΜΕΤΑΒΙΒΑΣΗ ΕΚΠΟΙΗΣΗ',
       'ΜΕΤΑΤΑΞΗ ΚΑΤΑΤΑΞΗ ΕΝΤΑΞΗ', 'ΠΡΟΘΕΣΜΙΑ', 'ΠΡΟΣΩΠΙΚΟ ΥΠΑΛΛΗΛΟΙ',
       'ΣΥΜΒΑΣΕΙΣ', 'ΣΥΜΜΟΡΦΩΣΗ ΔΙΟΙΚΗΣΕΩΣ', 'ΥΠΑΛΛΗΛΟΙ ΔΗΜΟΣΙΟΙ']

prob_thresh = (y_train_df.sum()/y_train_df.shape[0]).clip(upper=0.5)
prob_thresh

def gen_model_lemma():
  model = Sequential()
  model.add(Dense(vocab_len, activation='relu', input_shape=(50,)))
  model.add(Dropout(0.25))
  model.add(Dense(20, activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

lr_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)
epochs, batch_size = 10, 128

model = gen_model_lemma()
model.fit(x_train, y_train_df,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1,
          callbacks=[lr_reduction])

y_pred = model.predict(x_test)
predictions = pd.DataFrame(index=y_test_df.index, columns=y_test_df.columns)
for i in range(y_pred.shape[0]):
  predictions.iloc[i,:] = (y_pred[i,:]>prob_thresh).map({True:1, False:0})
print(accuracy(y_test_df, predictions))



#Νευρωνικό Μοντελοποίηση

n_in = 180
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
model.add(Dense(y_train.shape[1]))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='data/models/model-cnn_multi_filter.h5', save_best_only=True)
]

history = model.fit(x_train, y_train_df,
                    class_weight='auto',
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=callbacks)

model_cnn_multifilter = keras.models.load_model('data/models/model-cnn_multi_filter.h5')
metrics = model_cnn_multifilter.evaluate(x_test, y_test_df)

print("{}: {}".format(model_cnn_multifilter.metrics_names[0], metrics[0]))
print("{}: {}".format(model_cnn_multifilter.metrics_names[1], metrics[1]))

print ("\nΑναφορά Κατηγοριοποίησης")
y_pred = model_cnn_multifilter.predict(x_test)
predictions = pd.DataFrame(index=y_test_df.index, columns=y_test_df.columns)
for i in range(y_pred.shape[0]):
  predictions.iloc[i,:] = (y_pred[i,:]>prob_thresh).map({True:1, False:0})

accuracy(y_test_df, predictions)


#Μοντελοποίηση simple model
simple_model = Sequential()
simple_model.add(Embedding(5000, 100, input_length=x_train.shape[1]))
simple_model.add(Dropout(0.2))
simple_model.add(GlobalMaxPool1D())
simple_model.add(Dense(20 , activation='sigmoid'))
simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='data/models/model-simple.h5', save_best_only=True)
]
history_simple_model = simple_model.fit(x_train, y_train_df,
                    class_weight='auto',
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks)


simple_model = keras.models.load_model('data/models/model-simple.h5')
metrics = simple_model.evaluate(x_test, y_test_df)
print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))

print ("\nΑναφορά Κατηγοριοποίησης")
y_pred = simple_model.predict(x_test)
predictions = pd.DataFrame(index=y_test_df.index, columns=y_test_df.columns)
for i in range(y_pred.shape[0]):
  predictions.iloc[i,:] = (y_pred[i,:]>prob_thresh).map({True:1, False:0})
print(accuracy(y_test_df, predictions))

####################################################################
filter_length = 256
cnn_model= Sequential()
cnn_model.add(Embedding(5000, 100, input_length=x_train.shape[1]))
cnn_model.add(Dropout(0.3))
cnn_model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
cnn_model.add(GlobalMaxPool1D())
cnn_model.add(Dense(y_train_df.shape[1]))
cnn_model.add(Activation('sigmoid'))
cnn_model.summary()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='data/models/modelcnn.h5', save_best_only=True)
]
cnn_history = cnn_model.fit(x_train, y_train_df,
                    class_weight='auto',
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks)

cnn_model = keras.models.load_model('data/models/modelcnn.h5')
metrics = cnn_model.evaluate(x_test, y_test_df)

print("{}: {}".format(cnn_model.metrics_names[0], metrics[0]))
print("{}: {}".format(cnn_model.metrics_names[1], metrics[1]))

print ("\nΑναφορά Κατηγοριοποίησης")
y_pred = cnn_model.predict(x_test)
predictions = pd.DataFrame(index=y_test_df.index, columns=y_test_df.columns)
for i in range(y_pred.shape[0]):
  predictions.iloc[i,:] = (y_pred[i,:]>prob_thresh).map({True:1, False:0})
print(accuracy(y_test_df, predictions))


#lstm
lstm_out = 128
model_lstm = Sequential()
model_lstm.add(Embedding(5000, 128, input_length=x_train.shape[1]))
model_lstm.add(Bidirectional(LSTM(64)))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(y_train_df.shape[1],activation='softmax'))
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


lstm_history = model_lstm.fit(x_train, y_train_df,
                    class_weight='auto',
                    epochs=2,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks)

print ("\nΑναφορά Κατηγοριοποίησης LSTM")# lstm_model = keras.models.load_model('data/models/model-lstm.h5')
metrics = model_lstm.evaluate(x_test, y_test_df)

print("{}: {}".format(model_lstm.metrics_names[0], metrics[0]))
print("{}: {}".format(model_lstm.metrics_names[1], metrics[1]))
y_pred = model_lstm.predict(x_test)
predictions = pd.DataFrame(index=y_test_df.index, columns=y_test_df.columns)
for i in range(y_pred.shape[0]):
  predictions.iloc[i,:] = (y_pred[i,:]>prob_thresh).map({True:1, False:0})
print(accuracy(y_test_df, predictions))

######################################################################
'''Πρόβλεψη μιας καινούριας τυχαίας Γνωμοδότησης από το www.nsk.gr'''
######################################################################
q ='Επαναπρόσληψη απολυθέντων υπαλλήλων'
y_labels = mlb.classes_
def prediction_to_label(prediction):
    tag_prob = [(y_labels[i], prob) for i, prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))


tokenizer = Tokenizer(num_words=vocab_len, lower=True)
tokenizer.fit_on_texts(decisions_new.Concultatory)

def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=50)
f = get_features([q])


p1 = prediction_to_label(cnn_model.predict(f)[0])
p2 = prediction_to_label(model_lstm.predict(f)[0])
#p3 = prediction_to_label(lstm_model.predict(f)[0])
df_new = pd.DataFrame()
df_new['label'] = p1.keys()
df_new['p_cnn'] = p1.values()
df_new['model_lstm'] = p2.values()
#df_new['p_lstm'] = p3.values()
df_new['weighted'] = (2 * df_new['p_cnn'] + df_new['model_lstm']) / 3

df_new.sort_values(by='p_cnn', ascending=False)[:10]

