import argparse
import os
import numpy as np
import pickle
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('site', type=str)
parser.add_argument('--subsample', type=int, default=-1)
args = parser.parse_args()

site_name = args.site

label_defs = {'TP': 1, 'TN': 0, 'FN': 1, 'FP': 0}
pdb_dir = '/oak/stanford/groups/rbaltman/aderry/pdb/localpdb/mirror/pdb'

print(site_name)

emb_dir = '/scratch/users/aderry/prosite_embeddings'

with open(os.path.join(emb_dir, f'{site_name}.pkl'), 'rb') as f:
    data = pickle.load(f)
emb_all = data['embeddings']
labels_all = np.array([label_defs[lab] for lab in data['labels']], dtype=int)
pdbs_all = np.array(data['pdbs'])

fpfn_idx = (np.array(data['labels']) == 'FN') | (np.array(data['labels']) == 'FP')
tp_idx = (np.array(data['labels']) == 'TP')
tn_idx = (np.array(data['labels']) == 'TN')

emb_test = emb_all[fpfn_idx]
labels_test = labels_all[fpfn_idx]
pdbs_test = pdbs_all[fpfn_idx]

if args.subsample > 0:
    # num_sample = math.ceil(args.subsample * np.sum(tp_idx))
    num_sample = args.subsample
    print(f'Number of positive examples: {num_sample}')
    idx = np.flatnonzero(tp_idx)
    r = np.random.choice(idx, num_sample, replace=False)
    tp_idx[idx[~np.isin(idx, r)]] = False
    tp_idx_removed = np.zeros_like(tp_idx, dtype=bool)
    tp_idx_removed[idx[~np.isin(idx, r)]] = True
    emb_removed = emb_all[tp_idx_removed]
    labels_removed = labels_all[tp_idx_removed]

tptn_idx = tp_idx | tn_idx
emb_all = emb_all[tptn_idx]
labels_all = labels_all[tptn_idx]

svm_grid = {
    'C': [1, 10, 100, 500, 1000, 2500, 5000]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

svm = SVC(kernel='rbf', class_weight='balanced', probability=True)

tptn_scores = []
tptn_labels = []
models = []
for i, (train_idx, test_idx) in enumerate(outer_cv.split(emb_all, labels_all)):
    print('Fitting Fold', i)
    
    clf = GridSearchCV(svm, svm_grid, cv=inner_cv, scoring='average_precision', n_jobs=-1)
    emb_train = emb_all[train_idx]
    labels_train = labels_all[train_idx]
    emb_val = emb_all[test_idx]
    labels_val = labels_all[test_idx]
    if args.subsample > 0:
        emb_val = np.concatenate((emb_val, emb_removed))
        labels_val = np.concatenate((labels_val, labels_removed))
    
    clf.fit(emb_train, labels_train)
    tptn_scores.extend(list(clf.predict_proba(emb_val)))
    tptn_labels.extend(list(labels_val))
    models.append(clf.best_estimator_)
    print('\nBest Params:\n', clf.best_params_)
    print('\nBest Score:\n', clf.best_score_)
    
if args.subsample > 0:
    np.save(f'results/{site_name}_svm_{args.subsample}_y_prob', tptn_scores)
    np.save(f'results/{site_name}_svm_{args.subsample}_y_true', tptn_labels)
else:
    np.save(f'results/{site_name}_svm_y_prob', tptn_scores)
    np.save(f'results/{site_name}_svm_y_true', tptn_labels)


fold_scores = []
for model in models:
    fold_scores.append(model.predict_proba(emb_test))

if args.subsample > 0:
    np.save(f'results/{site_name}_svm_{args.subsample}_test_y_prob', fold_scores)
    np.save(f'results/{site_name}_svm_{args.subsample}_test_y_true', labels_test)
    np.save(f'results/{site_name}_svm_{args.subsample}_test_pdbs', pdbs_test)
else:
    np.save(f'results/{site_name}_svm_test_y_prob', fold_scores)
    np.save(f'results/{site_name}_svm_test_y_true', labels_test)
    np.save(f'results/{site_name}_svm_test_pdbs', pdbs_test)
