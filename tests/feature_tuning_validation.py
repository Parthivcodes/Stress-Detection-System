"""Synthetic-only feature and operating-point checks; never reads test respondents."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import pandas as pd
from feature_pipeline import FeatureEncoder,temperature
from tune_xgb_features import operating,sufficient
from tune_six_models import metric,decide

schema=json.loads(Path(sys.argv[1]).read_text())
keys=schema['numeric']+schema['categorical'];row={k:schema['limits'][k][0] for k in keys}
rows=[dict(row) for _ in range(3)]
for k in ['job_loss','bill_difficulty','utility_threat','transport_barrier']:rows[0][k]=2;rows[1][k]=1
rows[0].update(food_insecurity=5,isolation=5,support=1,sleep_duration=6,bmi=20)
rows[1].update(food_insecurity=1,isolation=1,support=5,sleep_duration=8,bmi=30)
rows[2].update(sleep_duration=10,bmi=40)
frame=pd.DataFrame(rows)
for mode,dim in [('original',69),('ordered',31),('engineered',38)]:
    enc=FeatureEncoder(schema,mode).fit(frame);x=enc.transform(frame)
    assert x.shape==(3,dim),(mode,x.shape)
    assert np.array_equal(x,FeatureEncoder.restore(enc.config()).transform(frame))
    if mode=='engineered':
        assert x[0,enc.names.index('social_risk_count')]==0
        assert x[1,enc.names.index('social_risk_count')]==7
        for i,k in enumerate(['sleep_under7','sleep_7to9','sleep_over9']):assert x[i,enc.names.index(k)]==1
        limits=enc.bmi_limits.copy();altered=frame.copy();altered.bmi=99.;enc.transform(altered);assert enc.bmi_limits==limits
rng=np.random.default_rng(42);p=rng.dirichlet([3,2,1],size=120);y=np.tile([0,1,2],40)
assert np.array_equal(temperature(p,2).argmax(1),p.argmax(1))
assert np.allclose(temperature(p,1),p)
reference=metric(y,p,decide(p,.2))
for ref in [None,reference]:
    found=operating(y,p,.6,ref);candidates=[]
    for t in np.unique(p[:,2]):
        m=metric(y,p,decide(p,t))
        if m['recall'][2]+1e-12>=.6:candidates.append((t,m))
    feasible=[(t,m) for t,m in candidates if ref is None or sufficient(m,ref,.6)]
    chosen=max(feasible or candidates,key=lambda item:(item[1]['accuracy'],item[0]))
    assert found['eligible']==bool(feasible)
    assert found['threshold']==chosen[0]
print('PASS: 69/31/38 feature dimensions, risk direction, sleep bands, frozen clipping, serialization, temperature, and exact threshold search versus brute force.')
