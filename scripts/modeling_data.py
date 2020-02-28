import pandas as pd
import os
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scripts.labelized import decisions_new

if os.path.exists('data/preprocessed/decisions_lemmas_train_preprocessed.csv') and os.path.exists(
        'data/preprocessed/decisions_lemmas_test_preprocessed.csv'):
    my_data_train = pd.read_csv('data/preprocessed/decisions_lemmas_train_preprocessed.csv')
    my_data_test = pd.read_csv('data/preprocessed/decisions_lemmas_test_preprocessed.csv')
else:
    my_data_raw = pd.read_csv('data/preprocessed/decisions_top40_lemmas.csv')
    min_test_size = 0.1
    my_data_train, my_data_test = train_test_split(my_data_raw, test_size=min_test_size, random_state=9)
    my_data_train.to_csv('./data/preprocessed/decisions_lemmas_train_preprocessed.csv', index=False)
    my_data_test.to_csv('data/preprocessed/decisions_lemmas_test_preprocessed.csv', index=False)

print('ποσοστό ΣΕΚ {0:.3f}'.format(my_data_test.shape[0]/decisions_new.shape[0]))
lemma_columns = ['ΑΔΕΙΑ ΙΔΡΥΣΕΩΣ ΛΕΙΤΟΥΡΓΙΑΣ', 'ΑΔΕΙΕΣ ΔΙΑΦΟΡΕΣ ΕΙΔΙΚΕΣ', 'ΑΕΙ ΔΕΠ',
       'ΑΚΙΝΗΤΑ', 'ΑΛΛΟΔΑΠΗ', 'ΑΝΑΚΛΗΣΗ', 'ΑΠΑΛΛΑΓΕΣ ΜΕΙΩΣΕΙΣ ΕΚΠΤΩΣΕΙΣ',
       'ΑΠΟΔΟΧΕΣ ΕΠΙΔΟΜΑΤΑ', 'ΑΠΟΖΗΜΙΩΣΗ', 'ΑΡΜΟΔΙΟΤΗΤΑ',
       'ΔΗΜΟΙ ΚΟΙΝΟΤΗΤΕΣ', 'ΔΗΜΟΣΙΑ ΕΡΓΑ', 'ΔΗΜΟΣΙΟ',
       'ΔΙΑΓΩΝΙΣΜΟΣ ΜΕΙΟΔΟΣΙΑΣ ΠΛΕΙΟΔΟΣΙΑΣ', 'ΔΙΕΥΘΥΝΤΗΣ ΠΡΟΙΣΤΑΜΕΝΟΣ',
       'ΔΙΟΡΙΣΜΟΣ ΠΡΟΣΛΗΨΗ', 'ΕΚΠΑΙΔΕΥΤΙΚΟΙ ΥΠΑΛΛΗΛΟΙ',
       'ΕΝΟΠΛΕΣ ΔΥΝΑΜΕΙΣ', 'ΕΠΙΤΡΟΠΕΣ ΔΙΑΦΟΡΕΣ', 'ΕΤΑΙΡΕΙΕΣ',
       'ΕΤΑΙΡΕΙΕΣ ΑΝΩΝΥΜΕΣ', 'ΙΑΤΡΟΙ', 'ΙΔΙΩΤΙΚΟ ΠΡΟΣΩΠΙΚΟ ΔΗΜΟΣΙΟΥ',
       'ΙΚΑ-ΕΤΑΜ', 'ΚΟΙΝΟΤΙΚΟ ΔΙΚΑΙΟ', 'ΚΡΑΤΙΚΕΣ ΠΡΟΜΗΘΕΙΕΣ',
       'ΜΕΤΑΒΙΒΑΣΗ ΕΚΠΟΙΗΣΗ', 'ΜΕΤΑΤΑΞΗ ΚΑΤΑΤΑΞΗ ΕΝΤΑΞΗ',
       'ΠΙΣΤΟΠΟΙΗΤΙΚΑ ΒΕΒΑΙΩΣΕΙΣ', 'ΠΟΛΕΟΔΟΜΙΑ ΡΥΜΟΤΟΜΙΑ', 'ΠΡΟΘΕΣΜΙΑ',
       'ΠΡΟΣΩΠΙΚΟ ΥΠΑΛΛΗΛΟΙ', 'ΠΡΟΥΠΗΡΕΣΙΑ ΧΡΟΝΟΣ ΥΠΗΡΕΣΙΑΣ', 'ΣΥΜΒΑΣΕΙΣ',
       'ΣΥΜΒΑΣΗ ΕΡΓΟΥ', 'ΣΥΜΜΕΤΟΧΗ', 'ΣΥΜΜΟΡΦΩΣΗ ΔΙΟΙΚΗΣΕΩΣ',
       'ΥΠΑΛΛΗΛΟΙ ΔΗΜΟΣΙΟΙ', 'ΦΟΡΟΛΟΓΙΑ ΕΙΣΟΔΗΜΑΤΟΣ',
       'ΧΟΡΗΓΗΣΗ ΑΝΤΙΓΡΑΦΩΝ ΕΓΓΡΑΦΩΝ']
sum_lemma = decisions_new[lemma_columns].sum()
sum_lemma_train = my_data_train[lemma_columns].sum()
sum_lemma_test = my_data_test[lemma_columns].sum()
df_decisions_per_lemma_1 = pd.DataFrame({'Lemma':lemma_columns, 'Total': sum_lemma_train/sum_lemma, 'Data Set': 'Train'})
df_decisions_per_lemma_2 = pd.DataFrame({'Lemma':lemma_columns, 'Total': sum_lemma_test/sum_lemma, 'Data Set': 'Test'})
df_decisions_per_lemma = pd.concat([df_decisions_per_lemma_1, df_decisions_per_lemma_2])

f, ax = plt.subplots(1, 1, figsize=(20, 10))
sns.barplot(data=df_decisions_per_lemma, x='Lemma', y='Total', hue='Data Set', axes=ax)
ax.set(ylabel='Αριθμός Γνωμ/σεων (Normalized)', xlabel='')
ax.set(title='Ποσοστό αριθμού γν/σεων για κάθε λήμμα στο ΣΕΚ,ΣΕΛ')
plt.xticks(rotation=90)
plt.show()

