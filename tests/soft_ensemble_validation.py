"""Runtime parity on validation rows only; no repeat holdout evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import model_paths
from src.ensemble_policy import classify,weighted_probabilities,ordinal_probabilities

parser=argparse.ArgumentParser()
parser.add_argument('--project',required=True)
parser.add_argument('--policy',required=True)
parser.add_argument('--compact',required=True)
args=parser.parse_args()
root=Path(args.project)
model_paths.PROJECT_ROOT=root; model_paths.MODEL_ROOT=root/'models'
os.environ['BRFSS_ENSEMBLE_DIR']=args.policy
os.environ['BRFSS_COMPACT_FOREST_DIR']=args.compact
from src.brfss_inference import BRFSSPredictor
np.testing.assert_allclose(ordinal_probabilities([.2,.8],[.4,.3]),[[.8,0,.2],[.2,.5,.3]])
np.testing.assert_array_equal(classify([[.4,.3,.3],[.35,.35,.3],[.36,.339,.301]],.301),[0,0,2])
try:
    weighted_probabilities({'x':np.array([.1,.2,.7])},{'x':.5})
    raise AssertionError('Bad weights accepted')
except ValueError: pass
engine=BRFSSPredictor('brfss2022_expanded')
assert engine.ensemble_policy and engine.ordinal_models is None
policy=engine.ensemble_policy
keys=engine.schema['numeric']+engine.schema['categorical']
frame=pd.read_csv(root/'data/expanded_survey_dataset/respondents.csv')
validation=frame.loc[frame.split=='validation'].iloc[::100].head(128)
max_error=0.0
for _,row in validation.iterrows():
    payload={k:float(row[k]) for k in keys}
    result=engine.predict(payload)
    engine.ensemble_policy=None
    old=engine.predict(payload)
    engine.ensemble_policy=policy
    for name in old['models']:
        a,b=old['models'][name],result['models'][name]
        assert a['predicted_class']==b['predicted_class']
        np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=0,atol=1e-6)
    raw={}
    for name in policy['weights']:
        # Invert the OLD display adjustment only for an independent check.
        p=np.array(old['models'][name]['probabilities'])/engine.selection[name]['multipliers']
        raw[name]=p/p.sum()
    expected=weighted_probabilities(raw,policy['weights'])
    actual=np.array(result['consensus']['probabilities'])
    max_error=max(max_error,float(np.max(np.abs(expected-actual))))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
    assert result['consensus']['class_id']==int(classify(actual,policy['high_threshold']))
    assert result['consensus']['method']=='weighted_raw_probabilities_with_high_cutoff'
    assert result['consensus']['weights']==policy['weights']
    assert result['consensus']['majority_baseline_accuracy']==20207/30718
print(json.dumps({'validation_requests':len(validation),'individual_model_classes_unchanged':True,
                  'max_probability_parity_error':max_error,'decision_boundary_tests':'passed'},indent=2))
