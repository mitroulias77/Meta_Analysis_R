import warnings
from imblearn.pipeline import Pipeline
from setuptools.command.test import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

from Reports.helper_functions import *
overall_f1_score_v1_cv = make_scorer(overall_f1_score_v1, greater_is_better=True)

my_data_train = pd.read_csv('data/preprocessed/decisions_lemmas_train_preprocessed.csv')
my_data_test = pd.read_csv('data/preprocessed/decisions_lemmas_test_preprocessed.csv')

train_X, train_y = my_data_train['Title'], my_data_train.drop(['Concultatory','Title'], axis=1)
test_X, test_y = my_data_test['Title'], my_data_test.drop(['Concultatory','Title'], axis=1)

lemma_columns = train_y.columns

prob_thresh = get_prob_thresh(my_data_train[lemma_columns],thresh_sel=1)

pipeline = Pipeline([
                ('cvec', CountVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
                ('clf', OneVsRestClassifier(LinearSVC(C=1, class_weight='balanced')))
            ])

pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))
br1_title = accuracy(test_y, predictions)
br1_title.to_excel('data/Results/br1_title.xlsx')
#Br2
pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))),
                    ('clf', OneVsRestClassifier(LogisticRegression(C=100)))
            ])
pipeline.fit(train_X, train_y)
print(accuracy(test_y, predictions))
br2_title = accuracy(test_y, predictions)
br2_title.to_excel('data/Results/br2_title.xlsx')
#br3

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5,min_df=2, ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(MultinomialNB(alpha = 0.01, fit_prior=False, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))


br3_title = accuracy(test_y, predictions)
br3_title.to_excel('data/Results/br3_title.xlsx')

#br4
pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5,min_df=2, ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(BernoulliNB(alpha = 0.005, fit_prior=False, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))
br4_title = accuracy(test_y, predictions)
br4_title.to_excel('data/Results/br4_title.xlsx')


pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5,min_df=2, ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(BernoulliNB(alpha = 0.005, fit_prior=False, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))


# predictions = best_clf.predict(test_X)

#Χρήση καλύτερων παραμέτρων απο την άπληστη αναζήτηση με 50% οριο πιθανότητας

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(MultinomialNB(alpha = 0.01, fit_prior=False, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))

#Χρήση Custom Συνάρτησης Πρόβλεψης
pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(MultinomialNB(alpha = 0.01, fit_prior=True, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)

prob, predictions = multi_label_predict(best_clf, test_X, prob_thresh)
print(accuracy(test_y, predictions))



#TFIDF + SVC
# LinearSVC --> Γραμμικό SVM και δεν εξάγει πιθανότητες

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
            ])
parameters = {
                'tfidf__max_df': (0.25, 0.5),
                'tfidf__ngram_range': [(1, 1),(1,2)],
                'tfidf__min_df': [1, 3],
                'clf__estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'clf__estimator__class_weight': ['balanced'],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4, verbose=10)
grid_search_cv.fit(train_X, train_y)

print()
print("Καλύτεροι παράμετροι συνόλου:")
print (grid_search_cv.best_estimator_.steps)
print()
best_clf = grid_search_cv.best_estimator_
predictions = best_clf.predict(test_X)
print(accuracy(test_y, predictions))

#individual hyperparameter

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.25, min_df = 2, ngram_range=(1, 2))),
                ('clf', OneVsRestClassifier(LinearSVC(C=1, class_weight='balanced', penalty='l2'), n_jobs=-1))
            ])
pipeline.fit(train_X,train_y)

predictions =pipeline.predict(test_X)
print(accuracy(test_y, predictions))

print("Accuracy = ",accuracy_score(test_y,predictions))
from scipy import stats
#TF-IDF + LR

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
grid_search_cv.fit(train_X, train_y)
best_clf = grid_search_cv.best_estimator_
# pipeline.fit(train_X, train_y)

prob, predictions = multi_label_predict(best_clf, test_X, prob_thresh)
print(accuracy(test_y, predictions))
br_titleLR = accuracy(test_y, predictions)
br_titleLR.to_excel('data/Results/br_titleLR.xlsx')

from sklearn.metrics import multilabel_confusion_matrix
matrix = multilabel_confusion_matrix(test_y, predictions)
print(matrix)
print(classification_report(test_y,predictions))
# predictions = best_clf.predict(test_X)
# accuracy(test_y,predictions)



# measuring performance on test set
print ("Εφαρμογή Βέλτιστου Ταξινόμητη στο σύνολο ελέγχου:")
best_clf = grid_search_cv.best_estimator_
prob, predictions = multi_label_predict(best_clf, test_X, prob_thresh)
print(accuracy(test_y, predictions))

# predictions = best_clf.predict(test_X)
# accuracy(test_y, predictions)
# A Single Hyperparameter
pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
                    ('clf', OneVsRestClassifier(LogisticRegression(C=100)))
            ])
pipeline.fit(train_X, train_y)

predictions = best_clf.predict(test_X)
accuracy(test_y, predictions)
prob, predictions = multi_label_predict(pipeline, test_X, prob_thresh)
print(accuracy(test_y, predictions))
#################################################################################
# CountVectorizer + SVC
# LinearSVC
pipeline = Pipeline([
            ('cvec', CountVectorizer()),
            ('clf', OneVsRestClassifier(LinearSVC())),
            ])
parameters = {
            'cvec__max_df': (0.25, 0.5,0.75, 1.0),
            'cvec__min_df': (1, 2),
            'cvec__ngram_range': [(1, 1), (1, 2)],
            'clf__estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "clf__estimator__class_weight": ['balanced']
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v1_cv)
grid_search_cv.fit(train_X, train_y)

print()
print("Best parameters set:")
print (grid_search_cv.best_estimator_.steps)
print()

# Μέτρηση Απόδοσης για το Σύνολο Ελέγχου
print ("Εφαρμογή Βέλτιστου Ταξινόμητη στο σύνολο ελέγχου:")
best_clf = grid_search_cv.best_estimator_
predictions = best_clf.predict(test_X)
print(accuracy(test_y, predictions))
br_cv_linear_svc = accuracy(test_y, predictions)
br_cv_linear_svc.to_excel('data/Results/br_cv_linear_svc.xlsx')

#TFIDF + SVC
# LinearSVC --> Γραμμικό SVM και δεν εξάγει πιθανότητες

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
            ])
parameters = {
                'tfidf__max_df': (0.25, 0.5),
                'tfidf__ngram_range': [(1, 1),(1,2)],
                'tfidf__min_df': [1, 3],
                'clf__estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'clf__estimator__class_weight': ['balanced'],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4, verbose=10)
grid_search_cv.fit(train_X, train_y)

print()
print("Καλύτεροι παράμετροι συνόλου:")
print (grid_search_cv.best_estimator_.steps)
print()
best_clf = grid_search_cv.best_estimator_
predictions = best_clf.predict(test_X)
print(accuracy(test_y, predictions))

br_cv_linear_svc = accuracy(test_y, predictions)
br_cv_linear_svc.to_excel('data/Results/br_tfidf_linear_svc.xlsx')

prob_thresh = get_prob_thresh(my_data_train[lemma_columns], thresh_sel=1)

pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)),
            ])
parameters = {
            'tfidf__max_df':[0.25,0.5,0.75],
            'tfidf__min_df':[1,2],
            'tfidf__ngram_range':[(1,1),(1,2)],
            'clf__estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=0)
grid_search_cv.fit(train_X, train_y)
print(best_clf = grid_search_cv.best_estimator_)

prob, predictions = multi_label_predict(best_clf, test_X, prob_thresh)
accuracy(test_y, predictions)

br_TFIDF_LR = accuracy(test_y, predictions)
br_TFIDF_LR.to_excel('data/Results/br_tfidf_LR.xlsx')



###################################


pipeline = Pipeline([
                ('cvec', CountVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))),
                ('clf',RandomForestClassifier(max_features='sqrt', n_jobs =4))
            ])

pipeline.fit(train_X, train_y)
predictions = pipeline.predict(test_X)
print(accuracy(test_y, predictions))


