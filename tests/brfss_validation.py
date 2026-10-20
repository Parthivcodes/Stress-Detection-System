"""Run from project root: python -m tests.brfss_validation or python tests/brfss_validation.py."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_paths import preserved_path
import numpy as np
import pandas as pd
from src.brfss_inference import BRFSSPredictor

frame=pd.read_csv(ROOT/'data/baseline_survey_dataset/respondents_30000.csv')
assert len(frame)==frame.respondent_id.nunique()==30000
engine=BRFSSPredictor('brfss2022');keys=engine.schema['numeric']+engine.schema['categorical']
assert frame.groupby(keys)['split'].nunique().max()==1
assert frame.groupby('respondent_id')['split'].nunique().max()==1
assert frame[keys].notna().all().all()
for row in frame.sample(12,random_state=7)[keys].to_dict('records'):
    result=engine.predict(row)
    assert len(result['models'])==6
    for m in result['models'].values():
        assert len(m['probabilities'])==3 and np.isclose(sum(m['probabilities']),1)
        assert all(0<=p<=1 for p in m['probabilities'])
good=frame.iloc[0][keys].to_dict()
for invalid in [{},None,{**good,'age':True},{**good,'age':17},{**good,'support':7},{**good,'bmi':float('nan')},{**good,'sex':1.5},{**good,'stress_raw':2}]:
    try:engine.predict(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted')
for path,h in json.loads((ROOT/'reports/brfss2022/legacy_hashes.json').read_text()).items():
    assert hashlib.sha256(preserved_path(path).read_bytes()).hexdigest()==h,path
print('PASS: 30,000 unique respondents, isolated splits, 72 real predictions, 8 invalid-input cases, original artifact hashes.')
