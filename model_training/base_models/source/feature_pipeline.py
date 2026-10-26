"""Versioned, JSON-serializable features and calibration; never edits source data."""
import numpy as np

ORDERED=['general_health','income','education','support','isolation','food_insecurity']

class FeatureEncoder:
    def __init__(self,schema,mode):
        if mode not in ('original','ordered','engineered'):raise ValueError(mode)
        self.schema=schema;self.mode=mode
    def fit(self,frame):
        self.mean=frame[self.schema['numeric']].mean().to_numpy().tolist()
        self.scale=frame[self.schema['numeric']].std(ddof=0).replace(0,1).to_numpy().tolist()
        self.bmi_limits=np.quantile(frame.bmi,[.01,.99]).tolist()
        self.transform(frame.iloc[:1]);return self
    def transform(self,frame):
        s=self.schema;features=s['numeric']+s['categorical']
        for key in features:
            lo,hi=s['limits'][key]
            if not np.isfinite(frame[key]).all() or not frame[key].between(lo,hi).all():raise ValueError('Invalid '+key)
            if key!='bmi' and not np.equal(frame[key],np.floor(frame[key])).all():raise ValueError('Noninteger '+key)
        a=frame[s['numeric']].to_numpy(dtype=float)
        if self.mode=='original':a=(a-np.asarray(self.mean))/np.asarray(self.scale)
        parts=[a];names=list(s['numeric'])
        for key in s['categorical']:
            lo,hi=s['limits'][key]
            if self.mode!='original' and key in ORDERED:
                parts.append(frame[[key]].to_numpy(dtype=float));names.append(key)
            else:
                categories=list(range(lo,hi+1))
                if self.mode!='original' and len(categories)==2:categories=categories[1:]
                for value in categories:
                    parts.append((frame[key].to_numpy()==value).astype(float)[:,None]);names.append(f'{key}_{value}')
        if self.mode=='engineered':
            # Survey codes: yes=1; isolation/food frequency 1=always ...5=never;
            # support reverses risk direction: 4=rarely,5=never.
            risks=sum((frame[k].to_numpy()==1).astype(int) for k in ['job_loss','bill_difficulty','utility_threat','transport_barrier'])
            risks+=(frame.food_insecurity.to_numpy()<=3)+(frame.isolation.to_numpy()<=3).astype(int)+(frame.support.to_numpy()>=4).astype(int)
            extra={'social_risk_count':risks,'unhealthy_14plus':(frame.unhealthy_days>=14).astype(int),'unhealthy_log1p':np.log1p(frame.unhealthy_days),'sleep_under7':(frame.sleep_duration<7).astype(int),'sleep_7to9':frame.sleep_duration.between(7,9).astype(int),'sleep_over9':(frame.sleep_duration>9).astype(int),'bmi_clipped':np.clip(frame.bmi,*self.bmi_limits)}
            for name,values in extra.items():parts.append(np.asarray(values,dtype=float)[:,None]);names.append(name)
        self.names=names
        return np.concatenate(parts,axis=1).astype('float32')
    def config(self):return dict(schema=self.schema,mode=self.mode,mean=self.mean,scale=self.scale,bmi_limits=self.bmi_limits,names=self.names)
    @classmethod
    def restore(cls,c):
        obj=cls(c['schema'],c['mode'])
        for key in ['mean','scale','bmi_limits','names']:setattr(obj,key,c[key])
        return obj

def temperature(p,t):
    if not np.isfinite(t) or t<=0:raise ValueError('Temperature must be positive.')
    z=np.log(np.clip(np.asarray(p,dtype=float),1e-12,1))/t
    z-=z.max(axis=1,keepdims=True);e=np.exp(z);return e/e.sum(axis=1,keepdims=True)
