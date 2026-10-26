"""Standalone diagnosis charts; optional temporary matplotlib dependency path."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,'/tmp/stress-diagnosis-20261026')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tmp'/'diagnosis_plot_windows'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp'/'diagnosis_matplotlib_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run(out):
    data=json.loads((out/'learning_curves.json').read_text());names=list(dict.fromkeys(r['model'] for r in data))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    aggregates=[]
    for n in names:
        for size in ['small','medium','full']:
            rr=[r for r in data if r['model']==n and r['size']==size]
            aggregates.append(dict(model=n,size=size,rows=rr[0]['rows'],runs=len(rr),train_accuracy_mean=float(np.mean([r['train']['raw']['accuracy'] for r in rr])),validation_accuracy_mean=float(np.mean([r['validation']['raw']['accuracy'] for r in rr])),validation_accuracy_std=float(np.std([r['validation']['raw']['accuracy'] for r in rr],ddof=1)) if len(rr)>1 else None,time_caps=sum(r['fit']['time_cap_reached'] for r in rr),best_epochs=[r['fit']['best_epoch'] for r in rr]))
    for measure,label,scale in [('accuracy','Raw argmax accuracy (%)',100),('logloss','Unweighted log loss',1)]:
        fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for ax,n in zip(axs.flat,names):
            for part,color in [('train','#2563eb'),('validation','#c2410c')]:
                xx=[];yy=[];sd=[]
                for size in ['small','medium','full']:
                    rr=[r for r in data if r['model']==n and r['size']==size];xx.append(rr[0]['rows'])
                    values=[r[part]['raw']['accuracy'] if measure=='accuracy' else r[part]['logloss'] for r in rr];yy.append(np.mean(values)*scale);sd.append((np.std(values,ddof=1) if len(values)>1 else 0)*scale)
                ax.plot(xx,yy,'o-',label=part,color=color);ax.fill_between(xx,np.array(yy)-sd,np.array(yy)+sd,color=color,alpha=.12)
            ax.set_title(n);ax.set_xlabel('Fitting respondents');ax.set_ylabel(label);ax.grid(alpha=.2);ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
            capped=sum(r['fit']['time_cap_reached'] for r in data if r['model']==n)
            if capped:ax.text(.03,.03,f'{capped} time-limited runs',transform=ax.transAxes,fontsize=8)
        axs[0,0].legend();fig.suptitle('Six-model learning curves: fixed development and reused-validation sets\nShading = ±1 seed standard deviation (neural: 2 seeds; classical: 1 seed)',fontsize=13)
        fig.savefig(out/f'learning_curves_{measure}.png',dpi=160);fig.savefig(out/f'learning_curves_{measure}.pdf');plt.close(fig)
    b=json.loads((out/'baseline.json').read_text());fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);x=np.arange(len(names));width=.36
    ax.bar(x-width/2,[b['models'][n]['train_diagnostic']['raw']['accuracy']*100 for n in names],width,label='Training sample (in-sample)',color='#2563eb')
    ax.bar(x+width/2,[b['models'][n]['validation']['raw']['accuracy']*100 for n in names],width,label='Reused validation',color='#c2410c');ax.axhline(b['majority_validation_accuracy']*100,color='#444',ls='--',label='Always rarely (validation)');ax.set_xticks(x,names,rotation=12);ax.set_ylim(45,85);ax.set_ylabel('Raw argmax accuracy (%)');ax.set_title('Current models: training–validation gap');ax.legend(loc='upper right',fontsize=8);fig.savefig(out/'current_model_gaps.png',dpi=160);plt.close(fig)
    (out/'learning_curve_aggregates.json').write_text(json.dumps(aggregates,indent=2)+'\n');print('Charts and learning-curve aggregates saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(Path(a.output))
