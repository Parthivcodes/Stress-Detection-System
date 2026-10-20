"""Verify stored results, real-data preservation and serving/training parity."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_paths import preserved_path, model_dir
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score
from src.brfss_inference import BRFSSPredictor

report=ROOT/'reports/brfss2022_tuned'
for path,expected in json.loads((report/'preserved_hashes.json').read_text()).items():
    assert hashlib.sha256(preserved_path(path).read_bytes()).hexdigest()==expected,path
data=pd.read_csv(ROOT/'data/baseline_survey_dataset/respondents_30000.csv').set_index('respondent_id')
saved=pd.read_csv(report/'test_predictions.csv').set_index('respondent_id')
assert len(saved)==5998 and saved.index.is_unique
assert set(saved.index)==set(data.index[data.split=='test'])
assert np.array_equal(saved.target,data.loc[saved.index,'target'])
metrics=json.loads((model_dir('brfss2022_tuned')/'metrics.json').read_text())
engine=BRFSSPredictor('brfss2022_tuned');keys=engine.schema['numeric']+engine.schema['categorical']
for metric in metrics:
    name=metric['Model'];p=saved[[f'{name} probability {c}' for c in range(3)]].to_numpy()
    assert np.allclose(p.sum(1),1) and np.isfinite(p).all() and (p>=0).all()
    assert np.array_equal(p.argmax(1),saved[name])
    assert np.isclose(accuracy_score(saved.target,saved[name]),metric['Accuracy'])
    assert np.isclose(f1_score(saved.target,saved[name],average='macro'),metric['F1-Score (Macro)'])
    assert np.isclose(roc_auc_score(saved.target,p,multi_class='ovr'),metric['ROC-AUC (Macro)'])
    assert np.allclose(recall_score(saved.target,saved[name],average=None),metric['Recall by class'])
for respondent in saved.sample(12,random_state=55).index:
    result=engine.predict(data.loc[respondent,keys].to_dict())
    assert result['model_version']=='brfss2022_tuned'
    for name,m in result['models'].items():
        assert np.allclose(m['probabilities'],saved.loc[respondent,[f'{name} probability {c}' for c in range(3)]].to_numpy(dtype=float),atol=1e-5)
        assert m['predicted_class']==saved.loc[respondent,name]
print('PASS: CSV and original models unchanged; all six held-out metrics independently reproduced; 72 serving predictions match saved evaluation probabilities.')
