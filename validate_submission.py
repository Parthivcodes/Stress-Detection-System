"""Reproducible duplicate audit and nested group-aware classical-model comparison.
Does not overwrite the six deployed model artifacts or original benchmark CSV.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.preprocessing import load_and_preprocess_data, prepare_train_test_split

ROOT = Path(__file__).resolve().parent

def main():
    frame = load_and_preprocess_data(str(ROOT / 'data/Sleep_health_and_lifestyle_dataset.csv'))
    split = prepare_train_test_split(frame, val_size=.10)
    X = frame[split['feature_cols']].to_numpy(dtype=np.float32)
    y = frame.Stress_Category.to_numpy()
    # Exact matching predictors after normalization; targets never define groups.
    _, groups = np.unique(X, axis=0, return_inverse=True)
    train_keys = {tuple(row) for row in split['X_train']}
    val_keys = {tuple(row) for row in split['X_val']}
    test_keys = [tuple(row) for row in split['X_test']]
    raw = pd.read_csv(ROOT / 'data/Sleep_health_and_lifestyle_dataset.csv')
    raw_unique = len(raw.drop(columns=['Person ID', 'Stress Level']).drop_duplicates())
    audit = dict(rows=len(X), raw_unique_predictor_profiles=raw_unique,
                 normalized_unique_predictor_profiles=len(np.unique(groups)),
                 original_train_rows=len(split['X_train']), original_validation_rows=len(split['X_val']),
                 original_test_rows=len(test_keys),
                 test_rows_matching_train=sum(key in train_keys for key in test_keys),
                 test_rows_matching_train_or_validation=sum(key in train_keys | val_keys for key in test_keys),
                 class_counts=np.bincount(y).tolist(),
                 conflicting_label_groups=sum(len(set(y[groups == g])) > 1 for g in np.unique(groups)))
    (ROOT / 'reports/data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    specs = {
        'Dummy': (DummyClassifier(strategy='most_frequent'), None),
        'Logistic Regression': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)), {'logisticregression__C': [.1, 1, 10]}),
        'Random Forest': (RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42, n_jobs=1), {'max_depth': [4, 8, None], 'min_samples_leaf': [1, 3]}),
        'XGBoost': (XGBClassifier(n_estimators=120, max_depth=5, learning_rate=.08, subsample=.85, colsample_bytree=.85, eval_metric='mlogloss', random_state=42, n_jobs=1), {'max_depth': [2, 5], 'learning_rate': [.03, .08]}),
    }
    records = []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])
        for name, (estimator, grid) in specs.items():
            for mode in (['fixed', 'tuned'] if grid else ['fixed']):
                model = clone(estimator)
                params = {}
                if mode == 'tuned':
                    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=100 + fold)
                    model = GridSearchCV(model, grid, cv=inner, scoring='f1_macro', n_jobs=1)
                    model.fit(X[tr], y[tr], groups=groups[tr])
                    params = model.best_params_
                else:
                    model.fit(X[tr], y[tr])
                prediction = model.predict(X[te])
                records.append(dict(model=name, mode=mode, fold=fold, test_rows=len(te), accuracy=accuracy_score(y[te], prediction), macro_f1=f1_score(y[te], prediction, labels=[0, 1, 2], average='macro', zero_division=0), parameters=json.dumps(params)))
                print(f'{fold}: {name} {mode}: F1={records[-1]["macro_f1"]:.4f}', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(ROOT / 'reports/grouped_validation_folds.csv', index=False)
    summary = result.groupby(['model', 'mode']).agg(accuracy_mean=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std')).reset_index()
    summary.to_csv(ROOT / 'reports/grouped_validation_summary.csv', index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
