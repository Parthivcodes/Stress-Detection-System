"""Versioned experimental encoders and feature-position-preserving 1D CNN."""
import numpy as np
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class FeatureEncoder:
    def __init__(self,schema,mode='original'):self.schema=schema;self.mode=mode
    def fit(self,frame):
        s=self.schema;cats=[list(range(s['limits'][k][0],s['limits'][k][1]+1)) for k in s['categorical']]
        self.base=ColumnTransformer([('numeric',StandardScaler(),s['numeric']),('categorical',OneHotEncoder(categories=cats,sparse_output=False,handle_unknown='error'),s['categorical'])]).fit(frame)
        self.names=s['numeric']+s['categorical'];return self
    def transform(self,f):
        if self.mode=='original':return self.base.transform(f).astype('float32')
        if self.mode!='engineered':raise ValueError('Unknown encoder mode.')
        # Hypothetical equal-weight contextual aggregates; not a validated scale
        # and not the simulator's exact label equation. No target/ID is used.
        support=(f.support-1)/4;isolation=(5-f.isolation)/4;food=(5-f.food_insecurity)/4;health=(f.general_health-1)/4
        material=sum((f[k]==1).astype(float) for k in ['job_loss','bill_difficulty','utility_threat','transport_barrier','medical_cost'])
        short=np.maximum(0,8-f.sleep_duration)
        extras=np.column_stack([support,isolation,food,health,material,short,np.maximum(0,f.sleep_duration-9),np.log1p(f.unhealthy_days),f.unhealthy_days>=14,
          support+isolation+food,health+support+isolation+food+material/3+short/2,
          short+f.general_health+f.support+(6-f.isolation)+(6-f.food_insecurity)+2*material+f.unhealthy_days/10,
          material*(support+isolation),short*health])
        return np.concatenate([f[self.names].to_numpy(dtype=float),extras],axis=1).astype('float32')

class PositionConv1D(nn.Module):
    """CNN over fixed tabular feature positions; no temporal-history claim."""
    def __init__(self,input_dim,channels=16,hidden=64,dropout=.15):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(1,channels,3,padding=1),nn.BatchNorm1d(channels),nn.ReLU(),nn.Conv1d(channels,channels,3,padding=1),nn.BatchNorm1d(channels),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(channels*input_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,3));self.skip=nn.Linear(input_dim,3)
    def forward(self,x):return self.head(self.conv(x.unsqueeze(1)))+self.skip(x)
