from prepare_data import ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
COLORS=['#98a3ad','#31829a','#df9a3a','#264b79','#6a8b57','#8c6c98']
NAMES={'country_previous':'Country prior','last_observation':'Last answer','history3':'3-month summary','hist_gradient_boosting':'Tabular model','deepseek_demo':'DeepSeek: profile','deepseek_history':'DeepSeek: own history','deepseek_shuffled_history':'DeepSeek: other history','conditional_donor':'Conditional donor'}

def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',dpi=220);fig.savefig(ROOT/f'figures/{name}.pdf');plt.close(fig)

def main():
    d=pd.read_csv(ROOT/'results/metrics.csv');d=d[(d.scope=='pooled')&~d.weighted]
    methods=['deepseek_demo','deepseek_history','deepseek_shuffled_history','last_observation','history3','hist_gradient_boosting']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,t,label in zip(axes,['c1120','c1220','c6120'],['1-year inflation','3-year-ahead annual inflation','Nominal spending growth']):
        values=[d[(d.target==t)&(d.method==m)].mae.iloc[0] for m in methods]
        ax.barh(np.arange(6),values,color=COLORS);ax.set_yticks(np.arange(6),[NAMES[m] for m in methods] if ax==axes[0] else ['']*6);ax.invert_yaxis();ax.set_title(label);ax.set_xlabel('MAE (percentage points; lower is better)')
        for i,v in enumerate(values):ax.text(v+.05,i,f'{v:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,max(values)*1.15)
    fig.tight_layout();save(fig,'numeric_error')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,t,label in zip(axes,['c3010','c3110','c7010'],['Past financial situation','Expected financial situation','Emergency liquidity']):
        values=[d[(d.target==t)&(d.method==m)].accuracy.iloc[0] for m in methods]
        ax.barh(np.arange(6),values,color=COLORS);ax.set_yticks(np.arange(6),[NAMES[m] for m in methods] if ax==axes[0] else ['']*6);ax.invert_yaxis();ax.set_title(label);ax.set_xlabel('Exact-match accuracy');ax.set_xlim(0,1)
        for i,v in enumerate(values):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
    fig.tight_layout();save(fig,'categorical_accuracy')
    p=pd.read_pickle(ROOT/'results/paired_predictions.pkl');z=p[p.target=='c1120'];truth=z.drop_duplicates('case_id').truth.to_numpy()
    fig,ax=plt.subplots(figsize=(8,3.5))
    for name,values,c in [('Observed',truth,'#172d43')]+[(NAMES[m],z[z.method==m].prediction.dropna().to_numpy(),c) for m,c in [('deepseek_demo','#98a3ad'),('deepseek_history','#31829a'),('deepseek_shuffled_history','#df9a3a'),('history3','#6a8b57')]]:
        v=np.sort(values);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=name,color=c,lw=2)
    ax.set_xlim(-5,25);ax.set_xlabel('1-year expected inflation (%)');ax.set_ylabel('Empirical cumulative probability');ax.legend(fontsize=8,loc='lower right');ax.set_title('Population fit can survive identity shuffling')
    save(fig,'inflation_ecdf')
    c=pd.read_csv(ROOT/'results/correction.csv');c=c[c.label_n==55]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for method,color in [('deepseek_demo','#98a3ad'),('deepseek_history','#31829a'),('history3','#6a8b57'),('hist_gradient_boosting','#264b79')]:
        vals=[];cov=[]
        for t in ['c1120','c1220','c3110','c7010']:
            m=c[(c.target==t)&(c.method==method)].sort_values('wave');h=c[(c.target==t)&(c.method=='human_only')].sort_values('wave')
            vals.append((m.rmse.to_numpy()/h.rmse.to_numpy()).mean());cov.append(m.coverage95.mean())
        axes[0].plot(range(4),vals,'o-',label=NAMES[method],color=color);axes[1].plot(range(4),cov,'o-',color=color)
    labels=['Inflation 1y','Inflation 3y','Adverse outlook','Liquidity']
    for ax in axes:ax.set_xticks(range(4),labels,rotation=15);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(1,color='#888',ls='--',lw=1);axes[0].set_ylabel('RMSE / human-only RMSE');axes[0].set_title('55 labels in each fixed panel of 220');axes[0].legend(fontsize=8)
    axes[1].axhline(.95,color='#888',ls='--',lw=1);axes[1].set_ylabel('Empirical coverage of nominal 95% CI');axes[1].set_ylim(.75,1);axes[1].set_title('1,000 random label samples per month')
    fig.tight_layout();save(fig,'correction_tradeoff')
    if (ROOT/'results/qwen_metrics.csv').exists():
        q=pd.read_csv(ROOT/'results/qwen_metrics.csv');q=q[q.calibration=='temperature'];fig,axes=plt.subplots(1,3,figsize=(10,3))
        for ax,metric in zip(axes,['accuracy','nll','jsd_probability_mean']):
            ax.bar(q.model,q[metric],color=['#98a3ad','#31829a','#6a8b57']);ax.set_title(metric.replace('_',' '))
        fig.tight_layout();save(fig,'qwen_comparison')

if __name__=='__main__':main()
