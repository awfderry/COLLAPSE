import argparse
import os
import numpy as np
import pickle
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str)
parser.add_argument('--data_dir', type=str, default='../data/xin_functional_benchmarks')
args = parser.parse_args()

print('Training models for', args.task)

with open(os.path.join(args.data_dir, f'{args.task}_embeddings.pkl'), 'rb') as f:
    data = pickle.load(f)


svm = SVC(kernel='rbf', probability=True)
svm_grid = {
    'C': [10, 50, 100, 200, 300, 400, 500]
}

mlp = MLPClassifier(hidden_layer_sizes=(256,))
mlp_grid = {
    'learning_rate_init': [1e-3, 1e-4],
    'alpha': [1e-3, 1e-4, 1e-5]
}

scores = []
labels = []
models = []
fold_aucs = []
for fold, fold_data in data.items():
    print('\nFitting Fold', fold)
    X_test = fold_data['X']
    y_test = np.array(fold_data['y'])
    
    X_train = []
    y_train = []
    for fold_, fold_data_ in data.items():
        if fold_ != fold:
            X_train.append(fold_data_['X'])
            y_train.extend(fold_data_['y'])
    X_train = np.concatenate(X_train, 0)
    y_train = np.array(y_train)

    clf = GridSearchCV(svm, svm_grid, scoring='roc_auc', n_jobs=-1)
 
    clf.fit(X_train, y_train)
    scores.extend(list(clf.predict_proba(X_test)[:, 1]))
    labels.extend(list(y_test))
    models.append(clf.best_estimator_)
    fold_aucs.append(clf.best_score_)
    print('\nBest Params:\n', clf.best_params_)
    print('\nBest Score:\n', clf.best_score_)

auc = roc_auc_score(labels, scores)

print(f'Overall AUROC across all folds: {auc}')
print(f'Fold-level AUROC: {np.mean(fold_aucs)} +/- {np.std(fold_aucs)}')
