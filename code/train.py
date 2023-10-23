# Import packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from pprint import pprint
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import joblib
import gzip


df_train = pd.read_csv('data/Ask0729-fixed.txt', sep='\t', header=None, names=["label", "text"])
df_test = pd.read_csv('data/testSet-qualifiedBatch-fixed.txt', sep='\t', header=None, names=["label", "text"])

df_train['target'] = df_train.apply(lambda r: 1 if r['label']=='Yes' else 0, axis=1)
df_test['target'] = df_test.apply(lambda r: 1 if r['label']=='Yes' else 0, axis=1)

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("rbf_svm", SVC()),
    ]
)

parameter_grid = {
        "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
        "vect__min_df": (1, 3, 5, 10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        "vect__norm": ("l1", "l2"),
        'rbf_svm__C': [1, 10, 100, 1000], 
        'rbf_svm__gamma': [0.001, 0.0001, 'scale'], 
        'rbf_svm__kernel': ['rbf', 'linear', 'poly'],
        'rbf_svm__class_weight': ['balanced', None]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=100,
    random_state=0,
    n_jobs=2,
    verbose=1,
)

print("Performing grid search...")
print("Hyperparameters to be evaluated:")
pprint(parameter_grid)

t0 = time()
random_search.fit(df_train.text, df_train.target)
print(f"Done in {time() - t0:.3f}s")

print("Best parameters combination found:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

test_accuracy = random_search.score(df_test.text, df_test.target)
print(
    "Accuracy of the best parameters using the inner CV of "
    f"the random search: {random_search.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")

# Saving the best model
joblib.dump(random_search.best_estimator_, gzip.open('model/e2e_pipeline.dat.gz', "wb"))

# Loading the best model
load_model = joblib.load('model/e2e_pipeline.dat.gz')
best_clf_pipe=load_model

# Predicting the class
y_pred=best_clf_pipe.predict(df_test.text)

# Classification Report
print(classification_report(df_test.target,y_pred))

# Confusion Matrix
fig, axs = plt.subplots(figsize=(8,8))
ConfusionMatrixDisplay.from_estimator(best_clf_pipe, df_test.text, 
                                      df_test.target, values_format='d',
                                      display_labels=['No Intent','Intent'],
                                      cmap='plasma', ax=axs)

