from datetime import datetime
import os
import sqlite3
from os import path
import re
import nltk
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from tqdm import tqdm
from classification.utils import remove_emphasis
import greek_stemmer as gr_stemm
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
print("\nΛήμματα: {}".format(dataframe['Category'].values[500]))
print("="*215)
import pandas as pd
decision_4 = dataframe['Concultatory'].values[3500]
print(decision_4)
print("\nΛήμματα: {}".format(dataframe['Category'].values[3500]))
print("="*215)
#Στατιστικά Γνωμοδοτήσεων
feats_df = pd.DataFrame()
feats_df["Length_Title"] = dataframe['Title'].apply(lambda x: len(str(x))) #μήκος λέξεων τίτλου
feats_df["Length_Concultatory"] = dataframe['Concultatory'].apply(lambda x: len(str(x))) #μήκος λέξεων Γνωμοδότησης
#Σώσιμο συνόλου δεδομένου με βασικά χαρακτηριστικά
feats_df.to_csv('data/basic_features.csv', columns=feats_df.columns)
feats_df.head(5)
#Εξαγωγή στατιστικών χαρακτηριστικών των γνωμοδοτήσεων
feats_df.describe()
#Συνάρτηση πυκνότητας πιθανότητας
plt.figure(figsize=(15, 8))
plt.subplot(1,3,1)
sns.distplot([feats_df['Length_Title']], color = 'green', axlabel="Κατανομή του μήκους των τίτλων")
plt.subplot(1,3,2)
sns.distplot([feats_df['Length_Concultatory']], color = 'blue', axlabel="Κατανομή του μήκους των περιλήψεων των Γνωμοδοτήσεων ")
plt.show()

# lemma_data_lower= pd.DataFrame([x.lower() for x in lemma_data.Category])
# lemma_data_lower['Category'] = lemma_data_lower
# lemma_data.head()
# lemma_data_lower.head()

def rearange(x):
    x = str (x).lower()
    x = x.replace ("παρ.", "").replace ("εκείνη", "").replace ("′Δ/νση", "Διεύθυνση").replace ("(πλειοψ.)", "") \
        .replace ("απ'", "από").replace ("κλπ", " ").replace ("(Τριμελούς Επιτροπής)", "") \
        .replace ("Αντιπρόεδρος Εισηγήτρια:", "").replace ("Προεδρεύων:", "").replace ("Αντιπρόεδρος Εισηγητής:", "") \
        .replace ("υπ.", "").replace ("αριθμ.", "").replace ("δ.κ.κ", "").replace ("πδ", "").replace("(πλειοψ.)","")

    return x
#Συναρτήσεις προεπεξεργασίας του νομικού κειμένου
def remove_numbers(s):
    s = re.sub("\S*\d\S*", " ", s).strip()
    return (s)
from string import punctuation

def strip_punctuation(w):
    cleaned_text  = ''.join(c for c in w if c not in punctuation)
    return (cleaned_text)

def removeEmphasis(word):
    word = word.split()
    no_emphasis = [remove_emphasis(x) for x in word]
    return (no_emphasis)

def removePatterns(w):
    ' '.join (word.strip (punctuation) for word in w.split()
              if word.strip (punctuation))
    return (w)

#stemmer = gr_stemm.GreekStemmer()#κλήση αντικειμένου για stemmingdef rearange(x):
def rearange(x):
    x = str (x).lower ( )
    x = x.replace ("παρ.", "").replace ("εκείνη", "").replace ("′Δ/νση", "Διεύθυνση").replace ("(πλειοψ.)", "") \
        .replace ("απ'", "από").replace ("κλπ", " ").replace ("(Τριμελούς Επιτροπής)", "") \
        .replace ("Αντιπρόεδρος Εισηγήτρια:", "").replace ("Προεδρεύων:", "").replace ("Αντιπρόεδρος Εισηγητής:", "") \
        .replace ("υπ.", "").replace ("αριθμ.", "").replace ("δ.κ.κ", "").replace ("πδ", "").replace("(πλειοψ.)","")

    return x
#Συναρτήσεις προεπεξεργασίας του νομικού κειμένου
def remove_numbers(sentence):
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    return (sentence)
from string import punctuation

def strip_punctuation(sentence):
    cleaned_text  = ''.join(c for c in sentence if c not in punctuation)
    return (cleaned_text)

def removeEmphasis(sentence):
    sentence = sentence.split()
    no_emphasis = [remove_emphasis(x) for x in sentence]
    return (no_emphasis)

def removePatterns(sentence):
    ' '.join (word.strip (punctuation) for word in sentence.split()
              if word.strip (punctuation))
    return (sentence)

#stemmer = gr_stemm.GreekStemmer()#κλήση αντικειμένου για stemming
# κλήση συνόλου stopwords
stop_words = nltk.corpus.stopwords.words('greek')
newStopWords = [' προς ','του','της','και','την','η','των','το','να', 'από', 'με', 'που', 'δεν', 'για',
                'αυτού','όπως','αυτό','όμως','στους','οποίες','ούτε','οποιο',
                'αλλη','ομως','αλλης','εις','ηδη','τουτο','αυτους','αυτης','οτι','μεχρι',
                'καθε','οπως','εχει','αυτην','εκτος','οποιας','συνεπως','επομενως',
                'όπως', 'αυτό', 'όμως', 'στούς', 'υπό', 'άνω', 'πλειοψ', 'κατ', 'αυτής', 'όχι', 'γ', 'οποίες', 'ούτε',
                'οποιο','αυτές', 'πριν', 'πυς', 'αυτού', 'δια', 'στα', 'ανευ', 'κχαο', 'οποια',
                'υπερ', 'αυτος', 'εφοσον', 'εντος', 'ενος', 'οπου', 'αβγδ', 'αυτου', 'πδτος', 'εδτδ',
                'σαν', 'τουτου', 'μεταξυ','ειτε', 'χρονο', 'μεχρι', 'ανηκει', 'νομου', 'νομο', 'εκτος', 'οποιας', 'συνεπως', 'επομενως',
                'προκειμενου', 'εννοια', 'οποιοι', 'καθως', 'οτι', 'τυχον', 'ηδη',
                'δηλαδη', 'περα΄', 'αυτης', 'ητοι', 'εες', 'αλλη', 'ομως', 'αλλης', 'εις', 'ηδη', 'τουτο', 'υπαρχει',
                'χωρις', 'χωρίς', 'οποίου', 'οποιου', 'λόγω', 'λογω', 'μονο', 'εχουν', 'καθε', 'οπως', 'εχει', 'αυτην',
                'μπορει', 'οποιες', 'ανωτερω', 'στους', 'διοτι', 'ουτε', 'οχι', 'ναι', 'δυναται', 'δυνατη', 'σχετικη',
                'αρθρου', 'αρθρο', 'χωρις', 'οσον', 'αφου', 'μερους', 'βαση', 'ανωτερω', 'ειχε', 'δυο', 'απαιτειται',
                'μονο', 'πρεπει', 'μπορει', 'αυτες', 'πε', 'ανω', 'διοτι', 'συμφωνα', 'βασει', 'οταν', 'μονον',
                'αυτον','αυτους', 'ισχυ', 'ιδιου', 'ηταν', 'και','τους','για', 'τις','στις','υπ’','γι’'
                'υβετ','οποτε','ολες','εφ’','κατ’','υπ’','υπ΄αριθ','υπ΄αριθμ.'
                ]
stop_words.extend(newStopWords)
print(stop_words)



#συνδιασμός όλων των παραπάνω
preprocessed_documents = []

for document in tqdm(dataframe['Concultatory'].values):
    filtered_sentence = []
    document = rearange (document)
    document = remove_numbers (document)
    document = strip_punctuation (document)
    document = removePatterns (document)
    document = removeEmphasis (document)
    #document = [x.upper() for x in document]
    #document = [stemmer.stem(word) for word in document if not word in stop_words and len(word)>=2]
    for cleaned_words in document:
        if ((cleaned_words not in stop_words) and (len (cleaned_words) > 2)):
            word = cleaned_words.lower()
            filtered_sentence.append(word)
        else:
            continue
    document = rearange (document)
    document = " ".join (filtered_sentence)
    preprocessed_documents.append(document.strip())

dataframe['Title'] = preprocessed_documents
# dataframe['Concultatory'] = preprocessed_documents


print("Το μέγεθος των γνωμοδοτήσεων είναι : {}".format(len(preprocessed_documents)))
dataframe.head ()
dataframe.to_csv("data/preprocessed/decisions_lemmas.csv",mode = 'w', index=False)




