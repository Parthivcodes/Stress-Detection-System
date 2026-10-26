"""Small deterministic checks; no project test-set access."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

ns=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/tune_six_models.py'))
p=np.array([[.6,.1,.3],[.3,.3,.4],[.1,.4,.5],[.2,.6,.2],[.7,.1,.2]])
y=np.array([2,2,2,1,0])
t=ns['cutoff'](y,p,2/3)
assert t==.4
assert np.array_equal(ns['decide'](p,t),[0,2,2,1,0])
assert ns['metric'](y,p,ns['decide'](p,t))['recall'][2]==2/3
assert np.array_equal(ns['decide'](np.array([[.4,.4,.2],[.3,.3,.4]]),.4),[0,2])
groups=np.repeat(np.arange(60),2)
frame=pd.DataFrame({'profile_group':groups,'target':groups%3})
a,b=ns['partition'](frame,5,17)
assert not set(a.profile_group)&set(b.profile_group)
assert len(a)+len(b)==len(frame)
good={'accuracy':.7,'macro_f1':.55,'qwk':.52,'recall':[.8,.3,.65]}
base={**good,'accuracy':.69}
assert ns['qualifies'](good,base,base,.65)
assert not ns['qualifies']({**good,'recall':[.8,.3,.64]},base,base,.65)
assert not ns['qualifies']({**good,'macro_f1':.54},base,base,.65)
assert not ns['qualifies']({**good,'recall':[.8,.29,.65]},base,base,.65)
print('PASS: cutoff boundaries, grouped partition isolation, and non-degradation gates.')
