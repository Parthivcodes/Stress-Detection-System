"""Explain saved results without fitting or re-evaluating any model."""
import json
from pathlib import Path
import sys
import pandas as pd

root=Path(sys.argv[1])
read=lambda name:json.loads((root/name).read_text())
frozen=read('frozen_selection.json');test=read('test_report.json')
reasons={};details=[]
for name,row in test.items():
    if name=='baselines':continue
    v=frozen['ensemble'] if name=='Ensemble' else frozen['models'][name]
    candidate=v['selected']['metrics'] if name=='Ensemble' else v['candidate']['constrained']
    baseline=v['original_matched'];actual=v['original_actual'];failed=[]
    if candidate['accuracy']<=baseline['accuracy']+1e-12:failed.append('validation accuracy did not beat the matched-recall original')
    for metric in ['macro_f1','qwk']:
        if candidate[metric]+1e-12<baseline[metric]:failed.append('validation '+metric+' decreased')
    if candidate['recall'][1]+1e-12<baseline['recall'][1]:failed.append('validation sometimes recall decreased')
    if candidate['accuracy']+1e-12<actual['accuracy']:failed.append('validation accuracy below the deployed model')
    if candidate['recall'][2]+1e-12<v['floor']:failed.append('validation high recall below its floor')
    if row['paired_vs_matched']['accuracy_delta_95ci'][0]<=0:failed.append('paired test accuracy interval does not establish a positive improvement')
    if row['candidate_constrained']['recall'][2]+1e-12<max(.6,row['original_actual']['recall'][2]):failed.append('test high recall below the deployed-model/60% safety floor')
    reasons[name]=failed
    for version in ['original_actual','original_matched','candidate_raw','candidate_constrained']:
        m=row.get(version)
        if m:details.append(dict(model=name,decision=version,accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],qwk=m['qwk'],sometimes_recall=m['recall'][1],high_recall=m['recall'][2],high_detected=m['high_detected'],high_missed=m['high_missed'],macro_auc=m['macro_auc']))
(root/'rejection_reasons.json').write_text(json.dumps(reasons,indent=2)+'\n')
pd.DataFrame(details).to_csv(root/'detailed_comparison.csv',index=False)
cv=[]
for path in sorted((root/'cv').glob('*.json')):
    r=json.loads(path.read_text());m=r['constrained']
    cv.append(dict(model=r['model'],config=r['config_id'],fold=r['fold'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],qwk=m['qwk'],high_recall=m['recall'][2],sometimes_recall=m['recall'][1],seconds=r['fit']['seconds']))
pd.DataFrame(cv).to_csv(root/'trial_history.csv',index=False)
print(json.dumps(reasons,indent=2))
print('Completed CV fits:',len(cv))
print('Total CV fitting seconds:',round(sum(r['seconds'] for r in cv),1))
for name,item in frozen['models'].items():
    print(name,'final fit seconds',round(sum(s['fit']['seconds'] for s in item['seed_reports']),1),'artifact bytes',item['candidate']['fit']['checkpoint_bytes'])
