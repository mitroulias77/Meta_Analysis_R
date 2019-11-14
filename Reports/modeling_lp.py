import warnings
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

train_X, train_y = my_data_train['Concultatory'], my_data_train.drop(['Concultatory','Title'], axis=1)
test_X, test_y = my_data_test['Concultatory'], my_data_test.drop(['Concultatory','Title'], axis=1)

lemma_columns = train_y.columns
# K-MEANS CLUSTERING
print('Αριθμός Μοναδικών Συνδυασμών Λημμάτων = ', train_y.drop_duplicates().shape[0])

# Από τις μέγιστες δυνατές 2 ^ 40 ετικέτες, βλέπουμε ότι υπάρχουν μόνο 1534 μοναδικοί συνδυασμοί.
# Θα χρησιμοποιήσουμε κάποια τεχνική ομαδοποίησης Κ-μέσων για να δούμε πόσα από αυτές τις
# ομάδες μπορούμε να τα μειώσουμε
ss = []
ks = range(10, 100)
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=2)
    labels = kmeans.fit_predict(train_y)
    ss.append(kmeans.inertia_)
f, axes = plt.subplots(figsize=(8, 8))
axes.plot(ks, ss, marker='.')
axes.set(xlabel='K = Αριθμός_ΟΜΑΔΑΣ', ylabel = 'Sum Square Error = Inertia', title = 'Elbow Plot')
plt.show()

pca = PCA()
pca.fit(train_y)
f, axes = plt.subplots(figsize=(8, 8))
axes.plot(range(1, pca.n_components_+1), pca.explained_variance_, marker = '.')
axes.set(xlabel = 'PCA features', ylabel='Variance', title='Explained variance of PCA features')
plt.show()

ks = [50, 60, 65, 70, 75, 90, 100]
f1_score = []
thresh = 0.85
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=2)
    labels = kmeans.fit_predict(train_y)
    cluster_center = pd.DataFrame(columns=train_y.columns)
    for cluster_id in range(k):
        cluster_center.loc[cluster_id] = (kmeans.cluster_centers_[cluster_id] >= thresh) * 1

    for idx, col in enumerate(train_y.columns):
        max_idx = kmeans.cluster_centers_[:, idx].argmax()
        max_value = kmeans.cluster_centers_[:, idx].max()
        if max_value < thresh:
            cluster_center.loc[max_idx, col] = 1

    y_pred = pd.DataFrame(columns=train_y.columns, index=train_y.index)
    for idx in range(k):
        y_pred.loc[labels == idx, :] = cluster_center.loc[idx, :].values

    result = accuracy(train_y, y_pred)
    f1_score.append(result.loc['Avg/Total', 'F1-Score'])

f, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.barplot(x=ks, y=f1_score, axes=ax)
ax.set(ylabel='Overall F1 Score', xlabel='k (Αριθμός από Clusters)')
ax.set(title='Overall F1 Score ΣΕΚ (Απώλεια οφειλόμενη σε ομαδοποίηση με 1472 CLUSTERS)')
plt.yticks(list(np.arange(0, 1, 0.1)))
for idx, val in enumerate(f1_score):
    ax.text(idx-0.1, val + 0.01,  str(val), color='black', fontweight='bold')
plt.show()

#Η προσέγγιση αυτή λαμβάνει υπόψη μερικούς συσχετισμούς μεταξύ των λημμάτων.
# Εδώ αντιμετωπίζουμε κάθε έναν από τους μοναδικούς συνδυασμούς των λημματων που υπάρχουν στα
# δεδομένα εκπαίδευσης ως μια πιθανή κατηγορία. Ως εκ τούτου, μπορεί να υπάρχει
# η χειρότερη περίπτωση των 2 ^ n_lemmas αριθμός των τάξεων.

train_y_cluster_labels= train_y.groupby(list(lemma_columns)).ngroup()
cluster_center = train_y.copy(deep=True)
cluster_center['Labels']=train_y_cluster_labels
cluster_center = cluster_center.drop_duplicates()
cluster_center = cluster_center.reset_index().set_index(['Labels']).sort_index().drop('index', axis=1)

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
overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = cluster_center)

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_cluster_labels)

print()
print("Best parameters set:")
print (grid_search_cv.best_estimator_.steps)
print()

# measuring performance on test set
print ("Applying best classifier on test data:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, cluster_center)
accuracy(test_y, predictions)

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
                ('clf', LinearSVC(C=10, class_weight='balanced'))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)

####################
ks = [75]
f1_score = []
thresh = 0.85
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=2)
    labels = kmeans.fit_predict(train_y)
    train_y_cluster_labels = pd.Series(labels, index=train_y)
    cluster_center = pd.DataFrame(columns=train_y.columns)
    for cluster_id in range(k):
        cluster_center.loc[cluster_id] = (kmeans.cluster_centers_[cluster_id] >= thresh) * 1

    for idx, col in enumerate(train_y.columns):
        max_idx = kmeans.cluster_centers_[:, idx].argmax()
        max_value = kmeans.cluster_centers_[:, idx].max()
        if max_value < thresh:
            cluster_center.loc[max_idx, col] = 1

    train_y_lemma_labels = pd.DataFrame(columns=train_y.columns, index=train_y.index)
    for idx in range(k):
        train_y_lemma_labels.loc[labels == idx, :] = cluster_center.loc[idx, :].values

    result = accuracy(train_y, train_y_lemma_labels)

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LinearSVC(class_weight='balanced'))
            ])
# sorted(pipeline.get_params().keys()) # -- to obtain the GridSearchCV parameter names
parameters = {
                'tfidf__max_df': [0.25, 0.5, 0.75],
                'tfidf__ngram_range': [(1, 2)],
                'tfidf__min_df': [1, 2, 5],
                'clf__C': [1, 10, 100]
            }
overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = cluster_center)

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_cluster_labels)

print()
print("Καλύτερες Παράμετροι από το ΣΕΚ:")
print (grid_search_cv.best_estimator_.steps)
print()

print ("Προσαρμογή καλύτερου εκτιμητή στο ΣΕΛ:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, cluster_center)
accuracy(test_y, predictions)

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))),
                ('clf', LinearSVC(C=1, class_weight='balanced'))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)


pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
                ('clf', LinearSVC(C=10, class_weight='balanced'))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)



# TF-IDF + RandomForest Classifier

train_y_cluster_labels = pd.DataFrame(columns=['Labels'], index=train_y.index)
train_y_cluster_labels['Labels'] = train_y.groupby(list(lemma_columns)).ngroup()
cluster_center = pd.DataFrame(train_y)
cluster_center['Labels']=train_y_cluster_labels
cluster_center = cluster_center.drop_duplicates()
cluster_center = cluster_center.reset_index().set_index(['Labels']).sort_index().drop('index', axis=1)


pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))),
                ('clf', RandomForestClassifier(n_estimators=500, max_depth=70, max_features='sqrt', n_jobs=4))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)

#Count Vectorizer + Linear SVC

# Using all (1472) lemma combinations

train_y_cluster_labels= train_y.groupby(list(lemma_columns)).ngroup()
cluster_center = train_y.copy(deep=True)
cluster_center['Labels']=train_y_cluster_labels
cluster_center = cluster_center.drop_duplicates()
cluster_center = cluster_center.reset_index().set_index(['Labels']).sort_index().drop('index', axis=1)


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
overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = cluster_center)
grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, verbose=10, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_cluster_labels)

print()
print("Καλύτερες Παράμετροι από το ΣΕΚ:")
print (grid_search_cv.best_estimator_.steps)
print()
# Μέτρηση Απόδοσης
print ("Προσαρμογή καλύτερου εκτιμητή στο ΣΕΛ:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, cluster_center)
accuracy(test_y, predictions)


pipeline = Pipeline([
                ('cvec', CountVectorizer(max_df=0.75, min_df=1, ngram_range=(1, 1))),
                ('clf', LinearSVC(C=1, class_weight='balanced'))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)


# TF-IDF + Naive Bayes
# Using all 1472 lemma Combinations
train_y_cluster_labels= train_y.groupby(list(lemma_columns)).ngroup()
cluster_center = train_y.copy(deep=True)
cluster_center['Labels']=train_y_cluster_labels
cluster_center = cluster_center.drop_duplicates()
cluster_center = cluster_center.reset_index().set_index(['Labels']).sort_index().drop('index', axis=1)

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

overall_f1_score_v2_cv = make_scorer(overall_f1_score_v2, greater_is_better=True, class_to_lemma_map = cluster_center)
grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, scoring=overall_f1_score_v2_cv)
grid_search_cv.fit(train_X, train_y_cluster_labels)

print()
print("Best parameters set:")
print (grid_search_cv.best_estimator_.steps)
print()

# measuring perfo
# rmance on test set
print ("Applying best classifier on test data:")
best_clf = grid_search_cv.best_estimator_
predictions = multi_class_predict(best_clf, test_X, cluster_center)
accuracy(test_y, predictions)

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))),
                ('clf', MultinomialNB(alpha=0.001, fit_prior=True, class_prior=None))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)


pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))),
                ('clf', MultinomialNB(alpha=0.001, fit_prior=True, class_prior=None))
            ])
pipeline.fit(train_X, train_y_cluster_labels)
predictions = multi_class_predict(pipeline, test_X, cluster_center)
accuracy(test_y, predictions)