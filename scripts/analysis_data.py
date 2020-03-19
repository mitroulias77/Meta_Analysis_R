from datetime import datetime
import os
import sqlite3
from os import path
import nltk
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud

nltk.download('wordnet')
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

file = path.join('data', 'nsk_decisions.csv')
data = pd.read_csv(file)
print("Στήλες Συνόλου Δεδομένων: ", [x for x in data.columns])
print("Γνωμοδοτήσεις: ", data.shape[0])

if not os.path.isfile('train.db'):
    disk_engine = create_engine('sqlite:///train.db')
    chunksize = 20000
    k = 0
    index_start = 1
    for df in pd.read_csv('data/nsk_decisions.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):
        df.index += index_start
        k+=1
        df.to_sql('data', disk_engine, if_exists='append')
        index_start = df.index[-1] + 1
# Απαρρίθμηση εγγραφών
if os.path.isfile('train.db'):
    start = datetime.now()
    con = sqlite3.connect('train.db')
    num_rows = pd.read_sql_query("""SELECT count(*) FROM data""", con)
    print("Αριθμός εγγραφών από τη Βάση Δεδομένων:",num_rows['count(*)'].values[0])
    con.close()
    print("Χρόνος εκτέλεσης εγγραφών:", datetime.now() - start)
# Έλεγχος διπλών εγγραφών
if os.path.isfile ('train.db'):
    start = datetime.now()
    con = sqlite3.connect ('train.db')
    no_dup_df = pd.read_sql_query ('SELECT Category,Concultatory,Title, COUNT(*) '
                                'AS duplicate_count FROM data GROUP BY Category,Concultatory,Title',con)
    con.close()

no_dup_df.head()
print('Αριθμός εγγραφών στο αρχικό Σύνολο: ',num_rows['count(*)'].values[0])
print('Αριθμός εγγραφών στο νέο Σύνολο: ', no_dup_df.shape[0])

#Αριθμός Γνωμοδοτήσεων που εμφανίζονται στη Βάση Δεδομένων
no_dup_df["lemma_count"] = no_dup_df["Category"].apply(lambda text: len(str(text).split(",")))
no_dup_df.head()
no_dup_df.lemma_count.value_counts()

#Κατανομή των λημμάτων
plt.figure(figsize=(12, 6))
sns.countplot(no_dup_df.lemma_count, palette ='gist_stern')
plt.title("Σύνολο Λημμάτων: Κατανομή του πλήθους εμφάνισης σε κάθε γνωμοδότηση")
plt.xlabel("Αριθμός Λημμάτων")
plt.ylabel("Πλήθος εμφάνισης στις Γνωμοδοτήσεις")
plt.show()
#Δημιουργία ΒΔ χωρίς διπλότυπα
if not os.path.isfile('no_duplicate.db'):
    engine = create_engine("sqlite:///no_duplicate.db")
    no_duplicate = pd.DataFrame(no_dup_df, columns=['Category', 'Concultatory', 'Title'])
    no_duplicate.to_sql('no_duplicate_train', engine)
if os.path.isfile('no_duplicate.db'):
    connection = sqlite3.connect('no_duplicate.db')
    lemma_data = pd.read_sql_query("""SELECT Category FROM no_duplicate_train""", connection)
    connection.close()
lemma_data.head()
# Ανάλυση τίτλου και περίληψης
con = sqlite3.connect('no_duplicate.db')
dataframe = pd.read_sql_query("""SELECT * FROM no_duplicate_train""", con)
con.close()
dataframe.head()
#Εκτύπωση τυχαίου δείγματος Γνωμοδοτήσεων
decision_1 = dataframe['Concultatory'].values[0]
print(decision_1)
print("\nΛήμματα: {}".format(dataframe['Category'].values[0]))
print("="*215)
decision_2 = dataframe['Concultatory'].values[100]
print(decision_2)
print("\nΛήμματα: {}".format(dataframe['Category'].values[100]))
print("="*215)
decision_3 = dataframe['Concultatory'].values[500]
print(decision_3)
print("\nCategory: {}".format(dataframe['Category'].values[500]))
print("="*215)

decision_4 = dataframe['Concultatory'].values[3500]
print(decision_4)
print("\nΛήμματα: {}".format(dataframe['Category'].values[3500]))
print("="*215)
#Στατιστικά Γνωμοδοτήσεων
feats_df = pd.DataFrame()
feats_df["Length_Title"] = dataframe['Title'].apply(lambda x: len(str(x))) #μήκος str τίτλου
feats_df["Length_Concultatory"] = dataframe['Concultatory'].apply(lambda x: len(str(x))) #μήκος str Γνωμοδότησης
#Σώσιμο συνόλου δεδομένου με βασικά χαρακτηριστικά
feats_df.to_csv('data/basic_features.csv', columns=feats_df.columns)
feats_df.head(5)
#Εξαγωγή στατιστικών χαρακτηριστικών των γνωμοδοτήσεων
feats_df.describe()
#Συνάρτηση πυκνότητας πιθανότητας
plt.figure(figsize=(15, 8))
plt.subplot(1,3,1)
sns.distplot([feats_df['Length_Title']], color = 'green', axlabel="Μηκος Χαρακτήρων Τίτλων")
plt.subplot(1,3,2)
sns.distplot([feats_df['Length_Concultatory']], color = 'blue', axlabel="Μήκος Χαρακτήρων Περιλήψεων")
plt.show()

lemma_data_lower= pd.DataFrame([x.lower() for x in lemma_data.Category])
lemma_data_lower['Category'] = lemma_data_lower
lemma_data.head()
lemma_data_lower.head()

#Συνολικός Αριθμός Μοναδικών Λημματών
def tokenize(m):
    m=m.split(',')
    lemmata=[i.strip() for i in m] #μερικά λήμματα περιέχουν κενά πρίν
    return lemmata

vectorizer = CountVectorizer(tokenizer = tokenize)
lemma_csr = vectorizer.fit_transform(lemma_data['Category'])
print("Σύνολο Γνωμοδοτήσεων :", lemma_csr.shape[0])
print("Μοναδικά Λήμματα :", lemma_csr.shape[1])

lemmas = vectorizer.get_feature_names()#επιστρέφει λεξικό.
print("Τα μοναδικά λήμματα είναι τα εξής :\n\n", lemmas[:2063])

#Αριθμός εμφάνισης λημμάτων
#https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
#Αποθηκεύουμε τα λήμματα από το dotmatrix σε λεξικό
freqs = lemma_csr.sum(axis=0).A1 #axis=0 στήλες. Που περιέχουν το πλήθος εμφάνιση
result = dict(zip(lemmas, freqs))

#Δημιουργία pandas dataframe από λεξικό
lemma_df = pd.DataFrame ({'Lemma': lemmas, 'Counts': freqs})
lemma_df.head()
#Ταξινόμηση ληματων σύμφωνα με τη συχνότητα εμφάνισης
lemma_df_sorted = lemma_df.sort_values(['Counts'], ascending=False)
lemma_counts = lemma_df_sorted['Counts'].values
lemma_df_sorted.head(100)

vectorizer = CountVectorizer(tokenizer=tokenize)
lemma_dtm = vectorizer.fit_transform(lemma_data['Category'])

print("Σύνολο Γνωμοδοτήσεων :", lemma_dtm.shape[0])
print("Μοναδικά Λήμματα :", lemma_dtm.shape[1])

# 'get_feature_name()' μας επιστρέφει λεξικό.
lemmas = vectorizer.get_feature_names()
# Αριθμός εμφάνισης λημμάτων
# https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
# Αποθηκεύουμε τα λήμματα από το dotmatrix σε λεξικό
freqs = lemma_dtm.sum(axis=0).A1  # axis=0 στήλες. Που περιέχουν το πλήθος εμφάνισης των λημμάτων
result = dict(zip(lemmas, freqs))

lemma_df = pd.DataFrame({'Lemma': lemmas, 'Counts': freqs})
# Ταξινόμηση ληματων σύμφωνα με τη συχνότητα εμφάνισης
lemma_df_sorted = lemma_df.sort_values(['Counts'], ascending=False)
lemma_counts = lemma_df_sorted['Counts'].values
# υπάλληλοι δημοσιοι , διορισμος πρόσλησψη, αρμοδιότητα, εταιρίες ανώνυμες, μεταταξη ...
# είναι τα πέντε λήμματα
lemma_df_sorted.head(100)
# o πίνακας lemma_counts απαρριθμεί τα λήμματα σε όλο το σύνολο
lemma_counts
# Κατανομή των λημμάτων
plt.figure(figsize=(12, 6))
plt.plot(lemma_counts)
plt.title("Σύνολο Λημμάτων: Κατανομή του πλήθους εμφάνισης σε κάθε γνωμοδότηση")
plt.grid()
plt.xlabel("Αριθμoί Λημμάτων")
plt.ylabel("Πλήθος εμφάνισης στις Γνωμοδοτήσεις")
plt.show()

# Quantile
plt.figure(figsize=(5, 8))
sns.boxplot(data=lemma_df_sorted)
plt.xlabel("Αριθμός Λημμάτων")
plt.ylabel("Αριθμός Γνωμοδοτήσεων")

# πάνω από 10
list_lemmas_grt_thn_10 = lemma_df_sorted[lemma_df_sorted.Counts > 10].Lemma
# Εκτύπωση Λίστας
print('{} Λήμματα εφανίζονται σε πάνω από 10 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_10)))

# πάνω από 50
list_lemmas_grt_thn_50 = lemma_df_sorted[lemma_df_sorted.Counts > 50].Lemma
# Εκτύπωση Λίστας
print('{} Λήμματα εφανίζονται σε πάνω από 50 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_50)))

# πάνω από 100
list_lemmas_grt_thn_100 = lemma_df_sorted[lemma_df_sorted.Counts > 100].Lemma
# Εκτύπωση Λίστας
print('{} Λήμματα εφανίζονται σε πάνω από 100 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_100)))

# πάνω από 200
list_lemmas_grt_thn_200 = lemma_df_sorted[lemma_df_sorted.Counts > 200].Lemma
# Εκτύπωση Λίστας
print('{} Λήμματα εφανίζονται σε πάνω από 200 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_200)))

# πάνω από 400
list_lemmas_grt_thn_500 = lemma_df_sorted[lemma_df_sorted.Counts > 500].Lemma
# Εκτύπωση Λίστας
print('{} Λήμματα εφανίζονται σε πάνω από 500 Γνωμοδοτήσεις'.format(len(list_lemmas_grt_thn_500)))

# Λήμμα με την συχνότερη εμφάνιση
print("Λήμμα(Κατηγορία) με τη συχνότερη εμφάνιση: {}".format(lemma_df_sorted.iloc[0][0]))
print("Το λήμμα [{}]: εμφανίζεται {} φορές".format(lemma_df_sorted.iloc[0][0], lemma_counts[0]))

# Λήμμα ανά Γνωμοδότηση
# Αποθήκευση του πλήθους λημμάτων για κάθε γνωμοδότηση στη λίστα 'lemma_count'
lemma_decision_count = lemma_dtm.sum(axis=1).tolist()

# Μετατροπή κάθε τιμής στο 'lemma_decision_count' σε int
lemma_decision_count = [int(j) for i in lemma_decision_count for j in i]
print('Συνολικά έχουμε {} εγγραφές.'.format(len(lemma_decision_count)))
print(lemma_decision_count[:500])

print("Μέγιστος Αριθμός Λημμάτων ανά Γνωμοδόηση: %d" % max(lemma_decision_count))
print("Ελάχιστος Αριθμός Λημμάτων ανά Γνωμοδόηση: %d" % min(lemma_decision_count))
print("M.O. Αριθμός Λημμάτων ανά Γνωμοδόηση: %f" % ((sum(lemma_decision_count) * 1.0) / len(lemma_decision_count)))

# Πόσες Γνωμοδοτήσεις έχουν μέχρι 3 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x <= 3, lemma_decision_count))
print(len(lemma_greater_than_avg_count))

# Πόσες Γνωμοδοτήσεις έχουν μέχρι 4 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x <= 4, lemma_decision_count))
print(len(lemma_greater_than_avg_count))

# Πόσες Γνωμοδοτήσεις έχουν μέχρι 5 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x <= 5, lemma_decision_count))
print(len(lemma_greater_than_avg_count))


# Πόσες Γνωμοδοτήσεις έχουν μέχρι 6 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x <= 6, lemma_decision_count))
print(len(lemma_greater_than_avg_count))


# Πόσες Γνωμοδοτήσεις έχουν μέχρι 7 λήματα
lemma_greater_than_avg_count = list(filter(lambda x: x <= 7, lemma_decision_count))
print(len(lemma_greater_than_avg_count))


# Ιστόγραμμα κατανομής λημμάτων
plt.figure(figsize=(10, 5))
sns.countplot(lemma_decision_count, palette='gist_rainbow')
plt.title("Κατανoμή λημμάτων ανά Γνωμοδότηση")
plt.xlabel("Λήμματα")
plt.ylabel("Γνωμοδοτήσεις")
plt.show()

# Εφαρμογή Word Cloud για τα πιο συχνά εμφανιζόμενα λήμματα γνωμοδοτήσεων
# Σχεδίαση από Word Cloud
# Μετατροπή 'result' λεξικού σε tuple
tup = dict(result.items())

# Αρχικοποίηση του WordCloud χρησιμοποιώντας συχνότητες εμφάνισης λημμάτων.
wordcloud = WordCloud(background_color='black', width=1600, height=800, ).generate_from_frequencies(tup)
fig = plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig("data/tag.png")
plt.show()
# Κατανομή των συχνά εμφανιζόμενων λημμάτων από τη συχνότητά τους
i = np.arange(start=25, stop=65, step=1)
lemma_df_sorted.head(100).plot(kind='bar', figsize=(15, 10), rot=90, color='green')
plt.title('Τελικό Σύνολο Λημμάτων')
plt.xticks(i, lemma_df_sorted['Lemma'])
plt.xlabel('Λήμματα ΝΣΚ')
plt.ylabel('Γνωμοδοτήσεις ΝΣΚ')
plt.show()
