import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from Reports.helper_functions import accuracy, overall_f1_score_v2, multi_class_predict
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

from helper_functions import *

my_data_train = pd.read_csv('data/preprocessed/decisions_lemmas_train_preprocessed.csv')
my_data_test = pd.read_csv('data/preprocessed/decisions_lemmas_test_preprocessed.csv')

train_X, train_y = my_data_train['Title'], my_data_train.drop(['Concultatory','Title'], axis=1)
test_X, test_y = my_data_test['Title'], my_data_test.drop(['Concultatory','Title'], axis=1)

lemma_columns = train_y.columns
print('Αριθμός Μοναδικών Συνδυασμών Λημμάτων = ', train_y.drop_duplicates().shape[0])
train_y_labels= train_y.groupby(list(lemma_columns)).ngroup()
y_labels_lut = train_y.copy(deep=True)
y_labels_lut['Labels']=train_y_labels
y_labels_lut = y_labels_lut.drop_duplicates()
y_labels_lut = y_labels_lut.reset_index().set_index(['Labels']).sort_index().drop('index', axis=1)

pipeline = Pipeline([
                ('cvec', CountVectorizer()),
                ('clf', LinearSVC(class_weight='balanced'))
            ])
# sorted(pipeline.get_params().keys()) # --  για το GridSearchCV , εξαγωγή των παραμέτρων του
parameters = {
                'cvec__max_df': [0.25, 0.5],
                'cvec__ngram_range': [(1, 1)],
                'cvec__min_df': [1, 2],
                'clf__C': [1, 10, 50, 100]
            }
overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = y_labels_lut)
grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_labels)

print()
print("Καλύτερες Παράμετροι από το ΣΕΚ:")
print (grid_search_cv.best_estimator_.steps)
print()
# Μέτρηση Απόδοσης
print ("Προσαρμογή καλύτερου εκτιμητή στο ΣΕΛ:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, y_labels_lut)
print(accuracy(test_y, predictions))
lp_cv_LSVC = accuracy(test_y,predictions)
lp_cv_LSVC.to_excel('data/Results/lp_cv_LSVC.xlsx')


#####################
# TF-IDF + Naive Bayes
pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB(fit_prior=True, class_prior=None))
            ])
# sorted(pipeline.get_params().keys()) # -- to obtain the GridSearchCV parameter names
parameters = {
                'tfidf__max_df': [0.25, 0.5, 0.75],
                'tfidf__ngram_range': [(1, 1)],
                'tfidf__min_df': [1, 2, 5, 10],
                'clf__alpha': [0.001, 0.01, 0.1, 1]
            }

overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = y_labels_lut)
grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_labels)

print()
print("Best parameters set:")
print (grid_search_cv.best_estimator_.steps)
print()

print ("Applying best classifier on test data:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, y_labels_lut)
print(accuracy(test_y, predictions))

lp_tfidf_NB = accuracy(test_y,predictions)
lp_tfidf_NB.to_excel('data/Results/lp_tfidf_NB.xlsx')

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LinearSVC(class_weight='balanced'))
            ])
# sorted(pipeline.get_params().keys()) # -- to obtain the GridSearchCV parameter names
parameters = {
                'tfidf__max_df': [0.5],
                'tfidf__ngram_range': [(1, 2)],
                'tfidf__min_df': [2],
                'clf__C': [5, 10, 20, 50, 100]
            }
overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = y_labels_lut)

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_labels)

print()
print("Best parameters set:")
print (grid_search_cv.best_estimator_.steps)
print()

# measuring performance on test set
print ("Applying best classifier on test data:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, y_labels_lut)
print(accuracy(test_y, predictions))

lp_tfidf_LSVC = accuracy(test_y,predictions)
lp_tfidf_LSVC.to_excel('data/Results/lp_tfidf_LSVC.xlsx')

# Logistic Regression
pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=-1)),
            ])
parameters = {
            'tfidf__max_df':[0.25,0.5,0.75,1.0],
            'tfidf__min_df':[1,2],
            'tfidf__ngram_range':[(1,1),(1,2)],
            'clf__estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__estimator__class_weight': ['balanced']
            }
print (grid_search_cv.best_estimator_.steps)
print()
grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_cv.fit(train_X, train_y_labels)
best_clf = grid_search_cv.best_estimator_
# pipeline.fit(train_X, train_y)

predictions = multi_class_predict(best_clf, test_X, y_labels_lut)
print(accuracy(test_y, predictions))

lp_tfidf_LR = accuracy(test_y,predictions)
lp_tfidf_LR.to_excel('data/Results/lp_tfidf_LR.xlsx')



