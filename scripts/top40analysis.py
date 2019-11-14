import unicodedata
import greek_stemmer as gr_stemm
import math
import missingno as msno
import warnings
import re
import nltk
from nltk import WordNetLemmatizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

preprocessed_data = False
if os.path.exists('data/preprocessed/decisions_top15_lemmas.csv'):
    dataframe=pd.read_csv("data/preprocessed/decisions_top15_lemmas.csv")
    preprocessed_data = True
print("Μέγεθος δεδομένων: ", dataframe.shape)


pd.set_option('display.max_columns',9)
dataframe.head(2)

dataframe.loc[1, 'Title'], dataframe.loc[1, 'Concultatory']

dataframe.info()
msno.matrix(dataframe)
#δεν υπάρχουν missing values και όλα τα λήμματα είναι σε int

#Aνάλυση Των Δεδομένων

# μετατροπή σε categorical
lemma_columns = dataframe.columns.drop(['Concultatory', 'Title'])
for col in lemma_columns:
    dataframe[col] = dataframe[col].astype('category')
# Αριθμός λημμάτων ανα γνωμ/ση
sum_lemma = dataframe[lemma_columns].sum()
df_lemmas_per_decision = pd.DataFrame(
    {'Lemmas':lemma_columns, 'Total': sum_lemma})

f,ax = plt.subplots(1, 1, figsize=(12,10))
sns.barplot(data=df_lemmas_per_decision, x= 'Lemmas', y= 'Total', axes = ax)
ax.set(ylabel = 'Πλήθος Αποφάσεων')
plt.xticks(rotation=90)
plt.show()

#Λήμματα ανα γνω/ση
sum_decisions = dataframe[lemma_columns].sum(axis=1)
df_lemmas_per_decision = pd.DataFrame({'title': dataframe.Title, 'Αριθμός Λημμάτων':sum_decisions}).groupby('Αριθμός Λημμάτων').count()
f, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.barplot(data=df_lemmas_per_decision, x=df_lemmas_per_decision.index, y='title', axes=ax)
ax.set(yscale='log', ylim=(1, 1e5),xlabel='ΑΡΙΘΜΟΣ ΛΗΜΜΑΤΩΝ / ΓΝΩΜΟΔΟΤΗΣΗ', ylabel='ΓΝΩΜΟΔΟΤΗΣΕΙΣ')
plt.xticks(rotation=90)
plt.show()

print('Κατά μ.ο, οι γνωμοδοτήσεις ταξινομούνται με {0:.2f} λήμματα'.format(sum_decisions.mean()))
print('Αριθμός γν/σεων με 5 λήμματα = {0}'.format(len(sum_decisions[sum_decisions==5])))


def remove_tags(sentence):
    html_tag = '<.*?>'
    cleaned_sentence = re.sub(html_tag, ' ',  sentence)
    return cleaned_sentence

def remove_accents(sentence):
    cleaned_sentence = unicodedata.normalize('NFD', sentence)
    cleaned_sentence = cleaned_sentence.encode('ascii', 'ignore')
    cleaned_sentence = cleaned_sentence.decode("utf-8")
    return cleaned_sentence

def remove_punctuation(sentence):
    cleaned_sentence = re.sub(r'[?|!|\'|"|#]', '', sentence)
    cleaned_sentence = re.sub(r'[,|.|;|:|(|)|{|}|\|/|<|>]|-', ' ', cleaned_sentence)
    cleaned_sentence = cleaned_sentence.replace("\n"," ")
    return cleaned_sentence

def keep_alpha(sentence):
    alpha_sentence = re.sub('[^a-z A-Z]+', ' ', sentence)
    return alpha_sentence


def lower_case(sentence):
    lower_case_sentence = sentence.lower()
    return lower_case_sentence

def stemming(sentence):
    stemmer = gr_stemm.GreekStemmer()
    stemmed_words = [stemmer.stem(word) for word in sentence.split()]
    stemmed_sentence=' '.join(stemmed_words)
    return stemmed_sentence

def lemmatize_words(sentence):
    lem = WordNetLemmatizer()
    lemmatized_words = [lem.lemmatize(word, 'v') for word in sentence.split()]
    lemmatized_sentence=' '.join(lemmatized_words)

def remove_stop_words(sentence):
    stop_words = nltk.corpus.stopwords.words('greek')
    stop_words.update(['υπ΄','ολες','οποιου','και','τουτου','οπως','τετοια','φδχ','οχι','κατ΄','οποιος','γσες','αυτην','τβ','ενα','φδχ','τουτου'])
    no_stop_words=[word for word in sentence.split() if word not in stop_words]
    no_step_sentence = ' '.join(no_stop_words)
    return no_step_sentence

def text_preprocess(sentence):
    pre_processed_sentence = remove_tags(sentence)
    pre_processed_sentence = remove_accents(pre_processed_sentence)
    pre_processed_sentence = remove_punctuation(pre_processed_sentence)
    pre_processed_sentence = keep_alpha(pre_processed_sentence)
    pre_processed_sentence = lower_case(pre_processed_sentence)
    pre_processed_sentence = stemming(pre_processed_sentence) # Use Lemmatize instead of stemming
    pre_processed_sentence = lemmatize_words(pre_processed_sentence)
    pre_processed_sentence = remove_stop_words(pre_processed_sentence)

    return pre_processed_sentence

if not preprocessed_data:
    dataframe['Concultatory'] = dataframe['Concultatory'].apply(text_preprocess)
    dataframe.to_csv('data/preprocessed/decisions_top50_lemmas.csv', index=False)

dataframe.describe()

#Word Cloud
'''Κοινές λέξεις που χρησιμοποιούνται για κάθε λήμμα γνωμ/σης'''

def save_wordcloud_plots(data, labels):
     for idx, col in enumerate(labels):
         wordcloud = WordCloud(max_font_size=50).generate(' '.join(data[data[col] == 1]['Concultatory']))
         ax = plt.figure(figsize=(9, 9)).add_subplot(1, 1, 1)
         ax.imshow(wordcloud)
         ax.axis("off")
         ax.set(title='Λήμμα Γνωμ/σης: {0}'.format(col))
         plt.savefig('./Images/results/wordcloud_{0}.png'.format(col), bbox_inches='tight')
         plt.close()


def save_wordcloud_subplots(data, labels):
     num_plot = 3
     fig_per_plot = 10
     num_cols = 2
     num_rows = math.ceil(fig_per_plot / num_cols)
     for idx, col in enumerate(lemma_columns):
         if idx % fig_per_plot == 0:
             fig = plt.figure(figsize=(14, 20))
         wordcloud = WordCloud(max_font_size=50).generate(' '.join(dataframe[dataframe[col] == 1]['Concultatory']))
         ax = fig.add_subplot(num_rows, num_cols, idx % fig_per_plot + 1)
         ax.imshow(wordcloud)
         ax.axis("off")
         ax.set(title='Λήμμα Γνωμ/σης: {0}'.format(col))
         if (idx + 1) % fig_per_plot == 0 or idx == len(lemma_columns) - 1:
             plt.savefig('./Images/results/wordcloud_part{0}.png'.format(1 + idx // fig_per_plot), bbox_inches='tight')
             plt.close()
             fig = plt.figure(figsize=(14, 20))

if not os.path.exists('Images/results/wordcloud_test.png'):
    save_wordcloud_plots(dataframe, lemma_columns)
if not os.path.exists('Images/results/wordcloud_part1.png'):
    save_wordcloud_subplots(dataframe, lemma_columns)

 # Plots περιλήψεων tokens
#

'''Ανάλυση Συσχετίσεων Λημμάτων(Correlation Analysis)'''
# Για παράδειγμα : αν μια γν/ση ανηκει 'προσωπικο υπάλληλοι' με ποια λήμματα συσχετίζεται
import numpy as np
# Heatmap
corr_matrix = (dataframe[lemma_columns].astype('int')).corr()
mask = np.array(corr_matrix)
mask[np.tril_indices_from(mask)] = False
corr_matrix = (100*corr_matrix//1)/10

fig = plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, mask=mask, annot=True, cbar=True, vmax=7, vmin=-7, cmap='RdYlGn')
plt.show()

#Multi-Lemma Distribution Plots
def save_lemma_pdf_given_lemma_plots(data, labels):
    for idx, col in enumerate(labels):
        df_sum_given_genre = pd.DataFrame({'Lemma':labels, 'Total': data[data[col]==1][labels].sum()})
        df_sum_given_genre = df_sum_given_genre.sort_values('Total', ascending=False).head(10)
        df_sum_given_genre['Total'] = df_sum_given_genre['Total']/df_sum_given_genre['Total'].max()
        ax = plt.figure(figsize=(9, 9)).add_subplot(1, 1, 1)
        sns.barplot(data=df_sum_given_genre, x='Lemma', y='Total', axes=ax)
        ax.set(title='Κατανομή για το λήμμα: {0}'.format(col), xlabel='', ylabel='Γνωμοδοτήσεις(Normalized)')
        plt.xticks(rotation=90)
        plt.savefig('./Images/results/lemma_pdf_given_{0}.png'.format(col), bbox_inches='tight')
        plt.close()

def save_lemma_pdf_given_lemma_subplots(data, labels):
    num_decision = 3
    fig_per_decision = math.ceil(len(lemma_columns)/num_decision)
    num_cols = 3
    num_rows = math.ceil(fig_per_decision/num_cols)
    for idx, col in enumerate(lemma_columns):
         if idx%fig_per_decision==0:
             fig = plt.figure(figsize=(20, 30))
         df_sum_given_lemma = pd.DataFrame({'Lemma':labels, 'Total': data[data[col]==1][labels].sum()})
         df_sum_given_lemma = df_sum_given_lemma.sort_values('Total', ascending=False).head(10)
         df_sum_given_lemma['Total'] = df_sum_given_lemma['Total']/df_sum_given_lemma['Total'].max()
         ax = fig.add_subplot(num_rows, num_cols, idx%fig_per_decision+1)
         sns.barplot(data=df_sum_given_lemma, x='Lemma', y='Total', axes=ax)
         ax.set(title='Κατανομή για το λήμμα: {0}'.format(col), xlabel='', ylabel='Γνωμοδοτήσεις(Normalized)')
         plt.xticks(rotation=90)
         if (idx+1)%fig_per_decision==0 or idx==len(lemma_columns)-1:
             plt.savefig('./Images/results/lemma_pdf_part{0}.png'.format(1+idx//fig_per_decision), bbox_inches='tight')
             plt.close()
             fig = plt.figure(figsize=(20, 30))

if not os.path.exists('Images/results/lemma_pdf_given_test.png'):
    save_lemma_pdf_given_lemma_plots(dataframe, lemma_columns)
if not os.path.exists('Images/results/lemma_pdf_part1.png'):
    save_lemma_pdf_given_lemma_subplots(dataframe, lemma_columns)

#το παρακάτω είναι plot για το jupyter notebook
# fig = plt.figure(figsize=(18, 60))
# num_cols = 3
# num_rows = math.ceil(len(lemma_columns)/num_cols)
# for idx, col in enumerate(lemma_columns):
#     df_sum_given_lemma = pd.DataFrame({'Lemma':lemma_columns, 'Total': dataframe[dataframe[col]==1][lemma_columns].sum()})
#     df_sum_given_lemma = df_sum_given_lemma.sort_values('Total', ascending=False).head(10)
#     df_sum_given_lemma['Total'] = df_sum_given_lemma['Total']/df_sum_given_lemma['Total'].max()
#     ax = fig.add_subplot(num_rows, num_cols, idx+1)
#     sns.barplot(data=df_sum_given_lemma, x='Lemma', y='Total', axes=ax)
#     ax.set(title='Κατανομή για το λήμμα: {0}'.format(col), xlabel='', ylabel='Γνωμοδοτήσεις(Normalized)')
#     plt.xticks(rotation=90)
#
# plt.tight_layout()
# plt.show()

'''ΛΗΜΜΑΤΑ ΠΟΥ ΕΧΟΥΝ ΔΟΘΕΙ ΣΕ ΕΝΑ ΛΗΜΜΑ'''


def save_numLemma_pdf_given_lemma_plots(data, labels):
     for idx, col in enumerate(labels):
         df_lemmas_per_decision = pd.DataFrame({'Total': data[data[col] == 1][labels].sum(axis=1)})
         df_numL_given_lemma = pd.DataFrame(df_lemmas_per_decision['Total'].value_counts().sort_index().head(10))
         df_numL_given_lemma['Total'] = df_numL_given_lemma['Total'] / df_numL_given_lemma['Total'].max()
         ax = plt.figure(figsize=(9, 9)).add_subplot(1, 1, 1)
         sns.barplot(data=df_numL_given_lemma, x=df_numL_given_lemma.index, y='Total', axes=ax)
         ax.set(title='Κατανομή για το λήμμα: {0} '.format(col), xlabel='',
                ylabel='Γνωμοδοτήσεις(Normalized)')
         plt.savefig('./Images/results/numLemma_pdf_given_{0}.png'.format(col), bbox_inches='tight')
         plt.close()


def save_numLemma_pdf_given_lemma_subplots(data, labels):
     num_plot = 3
     fig_per_decision= math.ceil(len(lemma_columns) / num_plot)
     num_cols = 3
     num_rows = math.ceil(fig_per_decision / num_cols)
     for idx, col in enumerate(lemma_columns):
         if idx % fig_per_decision == 0:
             fig = plt.figure(figsize=(20, 30))
         df_lemmas_per_decision = pd.DataFrame({'Total': data[data[col] == 1][labels].sum(axis=1)})
         df_numL_given_lemma = pd.DataFrame(df_lemmas_per_decision['Total'].value_counts().sort_index().head(10))
         df_numL_given_lemma['Total'] = df_numL_given_lemma['Total'] / df_numL_given_lemma['Total'].max()
         ax = fig.add_subplot(num_rows, num_cols, idx % fig_per_decision + 1)
         sns.barplot(data=df_numL_given_lemma, x=df_numL_given_lemma.index, y='Total', axes=ax)

         if (idx + 1) % fig_per_decision == 0 or idx == len(lemma_columns) - 1:
             plt.savefig('./Images/results/numLemma_pdf_part{0}.png'.format(1 + idx // fig_per_decision),
                         bbox_inches='tight')
             plt.close()
             ax.set(title='Κατανομή για το λήμμα: {0} '.format(col), xlabel='',
                ylabel='Γνωμοδοτήσεις(Normalized)')
             fig = plt.figure(figsize=(20, 30))

if not os.path.exists('Images/results/numLemma_pdf_given_Test.png'):
     save_numLemma_pdf_given_lemma_plots(dataframe, lemma_columns)
if not os.path.exists('Images/results/numLemma_pdf_part1.png'):
     save_numLemma_pdf_given_lemma_subplots(dataframe, lemma_columns)

#Για το jupyter noteboook
# fig = plt.figure(figsize=(18, 50))
# num_cols = 3
# num_rows = math.ceil(len(lemma_columns)/num_cols)
# for idx, col in enumerate(lemma_columns):
#     df_lemmas_per_decision = pd.DataFrame({'Total': dataframe[dataframe[col]==1][lemma_columns].sum(axis=1)})
#     df_numL_given_lemma = pd.DataFrame(df_lemmas_per_decision['Total'].value_counts().sort_index().head(10))
#     df_numL_given_lemma['Total'] = df_numL_given_lemma['Total']/df_numL_given_lemma['Total'].max()
#     ax = fig.add_subplot(num_rows, num_cols, idx+1)
#     sns.barplot(data=df_numL_given_lemma, x=df_numL_given_lemma.index, y='Total', axes=ax)
#     ax.set(title='Κατανομή για το λήμμα: {0} '.format(col), xlabel='',
#            ylabel='Γνωμοδοτήσεις(Normalized)')
# plt.tight_layout()
# plt.show()


