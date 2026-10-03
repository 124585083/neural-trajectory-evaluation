"""Scientific closeout figures, without picking cases from favorable scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trajectory_project.closeout_v1.closeout_setup import ROOT, ANIMALS, REPS


def plot_a(output_dir=None):
    plt.rcdefaults()  # Isolate from the descriptive module's explicit rcParams.
    output=Path(output_dir) if output_dir is not None else ROOT/'figures';output.mkdir(parents=True, exist_ok=True)
    a=pd.read_csv(ROOT/'results/A/self_reconstruction_main_table.csv')
    rr=pd.read_csv(ROOT/'results/A/round_self_metrics.csv')
    epochs=['full','hidden','no_bounce','post_bounce']
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex='col',layout='constrained')
    for col,(animal,rep) in enumerate((a,r) for a in ANIMALS for r in REPS):
        d=a[(a.animal==animal)&(a.representation==rep)&(a.coordinate=='y')].set_index('epoch').loc[epochs]
        for row,metric in enumerate(('r','RMSE')):
            ax=axes[row,col]
            for dx,head,color,label in ((-.13,'obj','#1769aa','Objective own target'),(.13,'beh','#db6d1c','Candidate own target')):
                ax.errorbar(np.arange(4)+dx,d[f'{metric}_{head}'],yerr=d[f'{metric}_{head}_sd'],
                            fmt='o',capsize=3,label=label,color=color,markersize=5)
            ax.set_xticks(range(4),['Full','Hidden','No bounce','Post bounce'],rotation=25)
            ax.set_ylabel('Pearson r' if row==0 else 'RMSE (position units)')
            ax.grid(axis='y',alpha=.2)
            if row==0:ax.set_title(f'{animal.title()} / {rep}')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('A: each ball path reconstructed against its own target\nMean ± SD of 100 overlapping condition splits; y coordinate')
    fig.savefig(output/'A_own_reconstruction.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,6),layout='constrained')
    for col,(animal,rep) in enumerate((a,r) for a in ANIMALS for r in REPS):
        d=rr[(rr.animal==animal)&(rr.representation==rep)&(rr.coordinate=='y')]
        for row,metric in enumerate(('Delta_r','Delta_RMSE')):
            ax=axes[row,col]
            for epoch,color in (('full','#527bbb'),('hidden','#dc9844')):
                ax.hist(d[d.epoch==epoch][metric],bins=20,histtype='step',lw=1.6,color=color,label=epoch)
            ax.axvline(0,color='black',ls='--',lw=.8)
            ax.set_xlabel('r_beh − r_obj (dimensionless)' if row==0 else 'RMSE_obj − RMSE_beh (position units)')
            ax.set_ylabel('Condition splits')
            if row==0:ax.set_title(f'{animal.title()} / {rep}')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Paired split differences: positive favors candidate own-target reconstruction\ny coordinate; 100 overlapping condition splits, not independent neural experiments')
    fig.savefig(output/'A_paired_differences.png',dpi=170);plt.close(fig)


def plot_nulls(output_dir=None):
    plt.rcdefaults()
    output=Path(output_dir) if output_dir is not None else ROOT/'figures'
    output.mkdir(parents=True, exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
    for col,(animal,rep) in enumerate((a,r) for a in ANIMALS for r in REPS):
        d=pd.read_csv(ROOT/'results/B'/f'{animal}_{rep}_q_aggregate.csv.gz')
        d=d[(d.coordinate=='y')&(d['head']=='D_beh')&(d.target=='behavior')]
        for row,metric in enumerate(('paired_r','paired_RMSE')):
            ax=axes[row,col]
            for epoch,color in (('full','#527bbb'),('hidden','#dc9844')):
                ax.hist(d[d.epoch==epoch][metric].dropna(),bins=30,histtype='step',lw=1.7,color=color,label=epoch)
            ax.axvline(0,color='black',ls='--',lw=.9)
            ax.set_xlabel('r_matched − r_shuffled (dimensionless)' if row==0 else 'RMSE_shuffled − RMSE_matched (position units)')
            ax.set_ylabel('Randomizations')
            if row==0:ax.set_title(f'{animal.title()} / {rep}')
    axes[0,0].legend(frameon=False)
    fig.suptitle('B: fixed candidate head, correct vs shuffled condition correspondence\ny coordinate; each statistic averages 100 splits on identical paired support; 1000 randomizations')
    fig.savefig(output/'B_paired_null_distributions.png',dpi=170);plt.close(fig)
    summary=pd.read_csv(ROOT/'results/C/random_endpoint_summary.csv')
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for col,(animal,rep) in enumerate((a,r) for a in ANIMALS for r in REPS):
        d=pd.read_csv(ROOT/'results/C'/f'{animal}_{rep}_per_q_mean100.csv')
        d=d[d.epoch=='full']
        sm=summary[(summary.animal==animal)&(summary.representation==rep)&(summary.epoch=='full')].set_index('metric')
        for row,metric in enumerate(('r','RMSE','skill')):
            ax=axes[row,col]
            ax.hist(d[metric].dropna(),bins=30,color='#a4b4c9',label='Random endpoint own target')
            ax.axvline(sm.loc[metric,'objective_mean'],color='#1769aa',lw=1.8,label='Objective own target')
            ax.axvline(sm.loc[metric,'behavior_mean'],color='#db6d1c',lw=1.8,ls='--',label='Actual candidate own target')
            if metric=='skill':ax.axvline(0,color='black',lw=.7,ls=':')
            ax.set_xlabel({'r':'Pearson r','RMSE':'RMSE (position units)','skill':'Skill vs mean-endpoint\ngeometry baseline'}[metric])
            if row==0:ax.set_title(f'{animal.title()} / {rep}')
            ax.set_ylabel('Endpoint allocations')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('C: full-interval own-target y scores after OLS refits\n1000 endpoint permutations; each statistic is the mean of 100 held-out splits')
    fig.savefig(output/'C_full_null_distributions.png',dpi=170);plt.close(fig)
    epochs=['full','hidden','no_bounce','post_bounce','endpoint_influence']
    fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for col,(animal,rep) in enumerate((a,r) for a in ANIMALS for r in REPS):
        for row,metric in enumerate(('r','RMSE','skill')):
            ax=axes[row,col]
            d=summary[(summary.animal==animal)&(summary.representation==rep)&(summary.metric==metric)].set_index('epoch').loc[epochs]
            ax.errorbar(np.arange(5),d.null_mean,yerr=d.null_sd,fmt='o',capsize=3,color='#737f90',label='Random: mean ± SD across q')
            ax.plot(np.arange(5)-.12,d.objective_mean,'o',color='#1769aa',label='Objective own target')
            ax.plot(np.arange(5)+.12,d.behavior_mean,'o',color='#db6d1c',label='Actual candidate own target')
            if row==0:ax.set_title(f'{animal.title()} / {rep}')
            if metric=='skill':ax.axhline(0,color='black',lw=.7,ls=':')
            ax.set_ylabel({'r': 'Pearson r (dimensionless)', 'RMSE': 'RMSE (position units)', 'skill': 'Skill (dimensionless)'}[metric])
            ax.set_xticks(range(5),['Full','Hidden','No bounce','Post bounce','Endpoint influence'],rotation=35,ha='right')
            ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('C: phase differences and mean-endpoint geometry baseline limits\nSame full-interval readout; y coordinate. Random mean ± SD across 1000 split-mean statistics.')
    fig.savefig(output/'C_phase_null_scores.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    plot_a()
    if (ROOT/'results/B/B_validation.json').exists() and (ROOT/'results/C/completed.json').exists():
        plot_nulls()
