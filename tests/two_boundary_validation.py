"""Synthetic checks only: exact lower-boundary optimization, ties and safety."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from tune_two_boundaries import boundaries,predict,cmstats,violations
from tune_six_models import metric
rng=np.random.default_rng(19);y=np.tile(np.arange(3),20)
p=rng.dirichlet([2,2,2],len(y));p[np.arange(len(y)),y]+=1;p/=p.sum(1,keepdims=True)
ref=metric(y,p);chosen=boundaries(y,p,ref)
hs=np.unique(np.r_[0,np.quantile(p[:,2],np.linspace(0,1,81)),1.000001])
qs=p[:,1]/(p[:,0]+p[:,1]);ls=np.r_[np.unique(qs),1.000001]
best=-1
for h in hs:
    for s in ls:
        pred=predict(p,dict(high=h,middle=s));cm=np.bincount(3*y+pred,minlength=9).reshape(3,3)
        acc,rec,f1,k=cmstats(cm)
        if rec[2]+1e-12>=max(.6,ref['recall'][2]) and rec[1]+1e-12>=ref['recall'][1] and f1+1e-12>=ref['macro_f1'] and k+1e-12>=ref['qwk']:best=max(best,acc)
assert chosen['feasible']==(best>=0)
if best>=0:
    assert abs(chosen['metrics']['accuracy']-best)<1e-12
    assert max(violations(chosen['metrics'],ref))<1e-12
ties=np.array([[.2,.2,.6],[.4,.4,.2],[.6,.2,.2]])
assert predict(ties,dict(high=.6,middle=.5)).tolist()==[2,1,0]
print('PASS: two-boundary optimization matches brute force; both recall constraints, F1/QWK, and tie rules checked.')
