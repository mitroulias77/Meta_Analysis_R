import warnings
warnings.filterwarnings("ignore")
import nltk
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import chain
import pandas as pd

dataframe=pd.read_csv("data/preprocessed/decisions_lemmas.csv")
dataframe.head()
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

dataframe['New_Lemmas'] = lemmata
decisions_new = dataframe[~(dataframe['New_Lemmas'].str.len() == 0)]

all_categories =  sum(lemmata,[])
len(all_categories)
len(set(all_categories))

decisions_new['Lemmata'] = decisions_new['New_Lemmas'].apply(lambda text : len(str(text).split(',')))

decisions_new.Lemmata.value_counts()
all_categories_new = nltk.FreqDist(all_categories)

count_lemmas = pd.DataFrame({'Lemma': list(all_categories_new.keys()),
                                 'Count': list(all_categories_new.values())})
count_lemmas_sorted = count_lemmas.sort_values(['Count'],ascending=False)
lemma_counts = count_lemmas_sorted['Count'].values

sorted_idx = count_lemmas_sorted.index
sorted_idx = list(sorted_idx)

lemmata_new = list(count_lemmas['Lemma'])
new_lemmata = [lemmata_new[word] for word in sorted_idx[25:65]]
new_categories = decisions_new['New_Lemmas']

for idx, row in decisions_new.iterrows():
    cats = row['New_Lemmas']
    new_cats = []
    for cat in cats:
        if cat in new_lemmata:
            new_cats.append(cat)
    decisions_new.at[idx, 'New_Lemmas'] = new_cats

columns=['index','Category','Lemmata']
decisions_new.drop(columns, axis=1, inplace=True)

decisions_new = decisions_new[decisions_new['New_Lemmas'].map(lambda d: len(d)) > 0]
decisions_new = decisions_new.reset_index(drop=True)


y = np.array([np.array(x) for x in decisions_new.New_Lemmas.values.tolist()])
mlb = MultiLabelBinarizer()
y_1= mlb.fit_transform(y)
y_labels = mlb.classes_

decisions_new = decisions_new.join(pd.DataFrame(mlb.fit_transform(decisions_new.pop('New_Lemmas')),
                          columns=mlb.classes_,
                          index=decisions_new.index))

decisions_new.to_csv("data/preprocessed/decisions_top40_lemmas.csv",mode = 'w', index=False)
# decisions_new.to_csv("data/preprocessed/decisions_top100_lemmas.csv",mode = 'w', index=False)
