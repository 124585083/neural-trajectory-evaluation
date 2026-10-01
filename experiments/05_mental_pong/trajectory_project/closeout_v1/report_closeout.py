"""Create the Chinese closeout narrative from the completed score files."""
import numpy as np
import pandas as pd
from trajectory_project.step3_io import read_json
from trajectory_project.closeout_v1.closeout_setup import ROOT, ANIMALS, REPS


def table(df, columns, digits=4):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return f'{v:.{digits}f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','/')
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in df[columns].itertuples(index=False,name=None)]
    return '\n'.join(lines)


def run():
    acceptance=read_json(ROOT/'results/final_acceptance_audit.json')
    assert acceptance['completed']
    a=pd.read_csv(ROOT/'results/A/self_reconstruction_main_table.csv')
    y=a[a.coordinate.eq('y')]
    full=y[y.epoch.eq('full')]
    cross=pd.read_csv(ROOT/'results/A/cross_2x2_summary.csv')
    cross=cross[cross.coordinate.eq('y')]
    b=pd.read_csv(ROOT/'results/B/B_summary.csv')
    b=b[b.coordinate.eq('y')]
    coverage=pd.read_csv(ROOT/'results/B/B_coverage_summary.csv')
    c=pd.read_csv(ROOT/'results/C/random_endpoint_summary.csv')
    d=read_json(ROOT/'results/descriptive/mean_cancellation_summary.json')
    geom=read_json(ROOT/'results/C/geometry_feasibility.json')
    owncols=['animal','representation','r_obj','r_beh','RMSE_obj','RMSE_beh','Delta_r','Delta_RMSE']
    matrix=[]
    for (animal,rep,epoch),g in cross[cross.epoch.isin(['full','hidden'])].groupby(['animal','representation','epoch'],sort=False):
        row=dict(animal=animal,representation=rep,epoch=epoch)
        for s in g.itertuples():row[s.cell]=f'{s.r:.4f} / {s.RMSE:.4f}'
        matrix.append(row)
    bp=b[b.epoch.isin(['full','hidden']) & b.is_self_target]
    bp=bp.pivot(index=['animal','representation','epoch','head','target'],columns='metric',values='mean').reset_index()
    cp=c[c.metric.isin(['r','RMSE','skill'])].pivot(index=['animal','representation','epoch'],columns='metric',
        values=['objective_mean','behavior_mean','null_mean','null_sd','behavior_empirical_random_at_least_as_good_fraction'])
    cp.columns=['_'.join(x) for x in cp.columns];cp=cp.reset_index()
    shortc=cp[cp.epoch.isin(['full','hidden','endpoint_influence'])]
    cov=coverage[coverage.metric.eq('shared_bins_fraction')].copy()
    cov['mean_shared_bin_fraction']=cov.mean_of_q_means
    rows=[]
    for animal,v in d['animals'].items():
        rows.append(dict(animal=animal,above=v['n_up_records'],below=v['n_down_records'],
            conditions_with_both_signs=v['n_conditions_with_both_error_signs'],
            mean_abs=v['condition_equal_mean_abs_trial_error'],abs_mean=v['condition_equal_abs_mean_error'],
            mean_square=v['condition_equal_mean_square_trial_error'],squared_mean=v['condition_equal_mean_squared_mean_error'],
            within_variance=v['condition_equal_mean_within_variance']))
    bfull=b[(b.epoch=='full')&(b['head']=='D_beh')&(b.target=='behavior')]
    bdr=bfull[bfull.metric=='paired_r']['mean']
    bde=bfull[bfull.metric=='paired_RMSE']['mean']
    cfull=cp[cp.epoch=='full']
    post=cp[cp.epoch=='post_bounce']
    skills=cfull[['objective_mean_skill','behavior_mean_skill','null_mean_skill']]
    assert (skills<0).all().all(), 'Revisit narrative rather than assume negative skills.'
    assert (full.Delta_r<0).all() and (full.Delta_RMSE<0).all()
    parts=[
    '# Mental-Pong condition-mean endpoint candidates: exploratory closeout',
    '## 1. Research question',
    'We first compare the reconstruction of each target. Check A evaluates the objective and candidate heads against their own and cross targets. Check B preserves each fitted head and changes test-condition correspondence. Check C preserves condition geometry, redistributes mean endpoints, and fits new OLS readouts. A has no random null distribution; B and C address different questions.',
    '**The representations contain readable task-position structure. Objective own-target reconstruction remains better in all four full-epoch groups. Actual candidates outperform randomized endpoint candidates in full, hidden, and endpoint-influence evaluations, with post-collision RMSE exceptions. Full-epoch neural readouts remain below the mean-endpoint geometry baseline; some Perle phases have slightly positive skill.**',
    '**raw_preprocessing=fail.** Prefix filtering and fit-source isolation pass within the published-input scope. Upstream filling across conditions and times remains unresolved. All conclusions retain this boundary.',
    '## 2. Data and final candidate definition',
    'Analyze the animals separately as condition-mean DMFC pseudopopulations, using half1. All79 physical IDs retain the original100 nominal39/40 splits. Condition59920 has an unknown collision anchor, leaving78 valid conditions and3369 condition-by-time rows per animal. Actual training/test counts can be38/39 and39/40, without replacement. Behavioral means use7407 Mahler and84873 Perle records; these are not independent neural trials.',
    'Read each stored FA50 and GPFA50 fitted on its39 training conditions. GPFA has one shared learnable RBF time scale. The50-dimensional intercept OLS has51 coefficients per coordinate and102 per xy head. No regularization, added scaling, behavioral features, or trial weights are introduced. Preserve completed50ms bin right edges and mixed/feedback masks. One mapping covers valid visible and hidden samples. Identical x labels and predictions provide an implementation check.',
    'The candidate uses the condition-mean final paddle endpoint b_bar. Collision paths follow the objective path before the anchor and connect that anchor to b_bar afterward. No-collision paths connect the initial position to b_bar. After the anchor, y=(1-a)y_anchor+a*b_bar, where a=(x-x_anchor)/(x_end-x_anchor). Preserve x, arrival time T, and support. The final version has no stop detection or repeated target updates.',
    '[Protocol and seeds](configs/closeout_protocol.json) · [Sources and hashes](SOURCES.md) · [Condition coverage](results/A/condition_label_coverage.csv).',
    '## 3. Executed work and version changes',
    'The [experiment ledger](experiment_ledger.md) traces paper reproduction, tool selection, early proxies, the earlier segmented candidates, trial endpoint labels, condition means, and A/B/C. GPFA64, Ridge, and small-subset findings retain their historical versions.',
    'The trial-label version repeated the same condition-mean neural input for different behavioral records. It did not validate single-trial neural prediction. The current version removes both within-condition target variation and the extra training/scoring weight of conditions with more trials. This is not a one-factor averaging ablation. See the [version comparison](results/A/previous_trial_version_comparison.csv).',
    '## 4. Own-target reconstruction',
    'Start with E_OO and E_BB. Delta_r=r_beh-r_obj; Delta_RMSE=RMSE_obj-RMSE_beh. Score original held-out condition-by-time rows in each split, then report mean/population SD across100 splits. Averaged prediction-curve scores do not replace this order. Position units are centered MWorks display coordinates.',
    table(full,owncols),
    f'Full-epoch candidate RMSE is higher by {(-full.Delta_RMSE).min():.4f}–{(-full.Delta_RMSE).max():.4f}, with lower correlation in all four groups. Hidden-phase results have the same direction. No-collision candidate RMSE is lower while correlation remains lower; post-collision candidate RMSE is higher. Full-epoch averages do not describe every phase identically.',
    table(y[y.epoch.isin(['hidden','no_bounce','post_bounce'])],['animal','representation','epoch','r_obj','r_beh','RMSE_obj','RMSE_beh','Delta_RMSE']),
    '![Own-target means and split SD](figures/A_own_reconstruction.png)',
    'The100 splits reuse the same conditions. SD describes split stability, not100 independent animal experiments. [All six phases and x](results/A/self_reconstruction_main_table.csv) · [Split scores](results/A/round_self_metrics.csv) · [Paired differences](figures/A_paired_differences.png).',
    '## 5. Complete2x2: target and readout distinctions',
    'OO=objective head to objective target; OB=objective head to candidate target; BO=candidate head to objective target; BB=candidate head to candidate target. Each cell reports r / RMSE.',
    table(pd.DataFrame(matrix),['animal','representation','epoch','OO','OB','BO','BB']),
    'Across all four full-epoch groups, OB outperforms BB for the same candidate target, and OO outperforms BO for the same objective target. Candidate-head training therefore provides no full-epoch held-out advantage. This differs from the OO/BB own-target comparison. Changing the reference for one fixed head can move correlation and RMSE in different directions; Mahler objective predictions have slightly lower RMSE against candidate labels but also lower correlation.',
    '[Five paired comparisons](results/A/contrast_summary.csv) · [All six2x2 phases](results/A/cross_2x2_summary.csv). Independent replay of400 pure-test prediction files differs by at most about1.1e-14 from original scores. OO/BB agree with own-target metrics and exclude training predictions.',
    'All79 condition tables and four-curve atlases remain [available](SOURCES.md). The [heterogeneity figure](figures/descriptive/all79_condition_heterogeneity.png) retains missing condition59920. IDs55062 and241919 were selected by the user after the earlier analysis and are illustrations, not independent confirmation. Their [current scores](results/descriptive/fixed_posthoc_case_scores.csv) and [cross-scores](results/descriptive/fixed_posthoc_case_2x2.csv) use this version.',
    '![Mahler fixed posthoc cases](figures/descriptive/mahler_fixed_posthoc_four_curves.png)',
    '![Perle fixed posthoc cases](figures/descriptive/perle_fixed_posthoc_four_curves.png)',
    'For55062, candidate own-target RMSE improves in both animals and representations, while correlation falls in Mahler and rises in Perle. For241919, candidate RMSE worsens in all four groups and correlation varies by animal. Head curves are generally close, but their separation is not always smaller than label separation: Perle241919 labels nearly coincide while mean predictions differ more. See the [case interpretation](results/descriptive/fixed_posthoc_case_interpretation.md).',
    'The four curves distinguish target separation from prediction separation. Small full-epoch bias can hide condition-specific offsets and amplitude compression. Shading denotes stability across splits where that condition was held out, not behavioral trial variation. Earlier stop-proxy improvements were not transferred into this version.',
    '## 6. B: fixed-head condition mismatch',
    'Each animal has1000 random repetitions over the original100 splits. Permute only valid test conditions. The same q/split mapping serves FA/GPFA and all four head-target combinations. Use equal timestamps valid for both conditions and belonging to the scored phase for both, then rescore matched predictions on that same intersection. Heads remain fixed. The table shows paired-support means for own-target cells; cross-target cells are also retained.',
    table(bp,['animal','representation','epoch','head','matched_r','null_r','paired_r','matched_RMSE','null_RMSE','paired_RMSE']),
    f'For B, paired_r=matched_r-null_r and paired_RMSE=null_RMSE-matched_RMSE; positive values favor correct correspondence. These effects differ from A own-target differences. Full-epoch BB paired_r is {bdr.min():.4f}–{bdr.max():.4f}; paired_RMSE is {bde.min():.4f}–{bde.max():.4f}. A short-support null cannot be subtracted directly from the original full-support score.',
    table(cov,['animal','epoch','mean_shared_bin_fraction','q025','q975','minimum_over_all_q_splits','maximum_over_all_q_splits']),
    'Shared fractions use each split original phase support as denominator. Quantiles summarize q means over100 splits; minima/maxima span all q-by-split combinations. Support varies with phase and mapping. Tail fractions are empirical random-control proportions, not exact fixed-support permutation p values. Each split has1000 distinct sampled permutations; individual conditions may retain their correspondence.',
    'Post-collision shared support averages about17.6% of bins and3.8 source conditions per evaluation. Among100000 q-by-split combinations,1955 Mahler and1894 Perle cases have zero shared support. They remain NA. Every q evaluates all100 splits; post-collision correlation averages92–100 finite splits, with counts retained per q. Full and hidden phases have no empty support. Phase nulls therefore do not establish results on complete original event-window support.',
    'Bounce-phase support is empty in1249 Mahler and1229 Perle cases. Finite split counts per q are94–100 and95–100. The other four phases have no empty support. All1000 q values per animal still have finite summaries. These100000 evaluations are not independent experiments.',
    '![Fixed-head paired random-control distributions](figures/B_paired_null_distributions.png)',
    '[Four-cell summaries and tails](results/B/B_summary.csv) · [Coverage](results/B/B_coverage_summary.csv) · [Mapping uniqueness](results/B/B_mapping_audit.csv) · [Mappings, common-bin counts, and split scores](artifacts/B/) · [Checks](results/B/B_validation.json). B supports condition/task correspondence while disrupting both physical and behavioral associations; it does not isolate behavioral representation.',
    '## 7. C: random endpoints with newly fitted OLS',
    'Permute each animal78 observed mean endpoints1000 times. Each allocation stays fixed across100 splits and the entire time axis, and is shared by FA/GPFA. Endpoint marginal distributions and ranges remain unchanged. All6084 geometric pairings per animal passed before random neural scoring, without clipping, rejection resampling, or anchor changes. On average98.8%/98.7% of identities change; all1000 allocations per animal are distinct.',
    'The400 actual sklearn multioutput OLS solves yield400000 equivalent random xy heads. Each random y target has51 coefficients; identical x targets share one equivalent solution. Batch results were checked against separate OLS and the original candidate head. Random heads see only training-condition random labels and are scored against their own held-out random paths. All coefficients, q-by-split scores, and predetermined q0/1/2 examples are saved.',
    'The table reports own-target r/RMSE. Null values average1000 statistics, each first averaged across its100 splits. Endpoint-influence support combines631 original post-collision bins and2162 no-collision bins after the initial point, totaling2793. This changes scoring support only; heads are not refitted by phase.',
    table(shortc,['animal','representation','epoch','objective_mean_r','behavior_mean_r','null_mean_r','behavior_mean_RMSE','null_mean_RMSE']),
    'Actual candidates outperform all1000 random-endpoint statistics on both r and RMSE in full, hidden, and endpoint-influence evaluations. The corresponding empirical count of random outcomes at least as good is0/1000. This is finite sampling resolution, not zero probability or independent-experiment significance.',
    'After collision, metrics diverge: correlation remains higher than the random reference, while RMSE has no consistent advantage.',
    table(post,['animal','representation','behavior_mean_r','null_mean_r','behavior_mean_RMSE','null_mean_RMSE','behavior_empirical_random_at_least_as_good_fraction_RMSE']),
    'Random candidates retain initial positions, collision anchors, x, and time structure; pre-collision samples equal the objective path. Random correlations therefore need not approach zero. Matched endpoint distributions do not ensure equal path distributions or reconstruction difficulty. [Phase-specific SD, range, objective separation, and errors](results/C/random_endpoint_summary.csv) are retained.',
    '![Full-epoch random-endpoint own-target distributions](figures/C_full_null_distributions.png)',
    'The mean-endpoint geometry baseline combines the mean endpoint of valid training conditions with each test condition known geometry. skill=1-SSE_neural/SSE_geometry; a zero denominator is NA. This baseline uses task geometry and is not a pure neural prediction. The objective baseline uses training objective endpoints while scores retain the original objective labels.',
    table(shortc,['animal','representation','epoch','objective_mean_skill','behavior_mean_skill','null_mean_skill']),
    '**Full-epoch skill is negative for all four groups.** Perle candidate skill is slightly positive for GPFA hidden samples(0.0288) and no-collision FA/GPFA(0.0623/0.1065). These phase exceptions remain explicit. Actual labels outperforming random endpoints support readable condition-endpoint correspondence under this randomization rule. They do not establish general performance beyond shared geometry; target difficulty also changes.',
    '![Phase differences and mean-endpoint geometry baseline](figures/C_phase_null_scores.png)',
    '[Complete random-endpoint summary](results/C/random_endpoint_summary.csv) · [Per-q/per-split scores](results/C/) · [Assignments, weights, and fixed examples](artifacts/C/) · [Geometry audit](results/C/geometry_feasibility.json) · [Completion and replay checks](results/C/completed.json).',
    '## 8. Integrated evidence',
    '**A. Readable task-position structure: supported within scope.** Held-out reconstruction, correct correspondence beating B mismatch, and actual targets beating C in the main ranges support this limited conclusion. They do not establish a true internal path, unique algorithm, or strict neural causal mechanism. The stronger geometry baseline limits performance claims beyond shared geometry.',
    '**B. Easier candidate reconstruction than objective reconstruction: unsupported for the full epoch.** OO outperforms BB in all four groups. Preserve no-collision RMSE exceptions and post-collision differences separately. Beating random targets cannot reverse the direct OO/BB result or convert different-target scores into a neural preference.',
    '**C. Different neural representations for opposite behaviors within a condition: untested.** Only condition-mean neural inputs are available, with unknown original trial members. This does not establish absence of such information.',
    '## 9. Averaging and preprocessing limits',
    'Using fixed behavioral members, define e_ci=b_ci-current_geometry_objective_endpoint and e_c=mean(e_ci). Signs use exact numerical comparison; no neural outcome sets a near-correct threshold. Both animals have zero exact-zero records. Means and variances below give equal weight to conditions.',
    table(pd.DataFrame(rows),['animal','above','below','conditions_with_both_signs','mean_abs','abs_mean','mean_square','squared_mean','within_variance']),
    'mean_square=squared_mean+within_variance, with maximum residual8.9e-15. Opposing errors cancel in most conditions; median cancellation is54.0% for Mahler and48.0% for Perle. Full-path candidate-objective RMS separation is0.6760/0.5612, and hidden separation1.0006/0.8472, below neural reconstruction RMSE. This describes reduced target separation after averaging; it does not fully explain the missing overall candidate advantage.',
    '[Condition-level counts and variability](results/descriptive/endpoint_condition_audit.csv) · [Member errors](results/descriptive/endpoint_member_errors.csv.gz) · [Path separation and reconstruction error](results/descriptive/path_separation_vs_error.csv) · [Interpretation](results/descriptive/mean_cancellation_interpretation.md). Path separation pools valid label rows; reconstruction RMSE is scored per held-out split before averaging. This is a magnitude comparison.',
    '![Descriptive cancellation in behavioral means](figures/descriptive/endpoint_mean_cancellation.png)',
    'Unknown neural membership prevents confirmation that the neural mean contains exactly these behavioral records. raw_preprocessing=fail reflects upstream filling across conditions and times; OLS, mismatching, and endpoint randomization do not remove it. Strictly pre-feedback endpoint timing remains unverified. Collision anchors derive from50ms released paths. Repeated splits and randomizations add no neural trials.',
    '## 10. A future experiment requiring unavailable data',
    'The [future design](future_design.md) groups above-target, below-target, and near-correct trials within one physical condition using thresholds fixed from measurement precision and task tolerance. Match absolute error, session, trial count, and available units. Recover unit/session/trial responses, behavior, timestamps, and mean membership before building group neural means with common representations and independent evaluation. Failures do not directly identify internal judgment errors. Group means are not blind new-trial predictions; cross-session pseudopopulations are not single population decisions; reliability requires genuine trial halves.',
    'One mixed condition mean cannot determine its behavioral subgroup means. No grouped neural experiment, synthetic neural trials, data collection, or replacement-model search was performed. This limitation concerns the obtained and audited data.',
    '## 11. Closeout',
    'A cross-scoring,1000 B mismatches, and1000 C endpoint-refit repetitions are complete, each random repetition covering the original100 splits. Averaging, all-condition heterogeneity, posthoc cases, and historical versions are documented. Evidence supports readable condition-mean task position, not a full-epoch candidate advantage or within-condition single-trial neural-behavioral differences.',
    '**Scientific status: CLOSED_EXPLORATORY_WITH_LIMITATIONS.** [Original acceptance](results/final_acceptance_audit.json), [execution routes](README.md), and [manifest](manifest.json) trace this completed study. Integration records separately describe publication edits; preserved original files retain their original checksums.'
    ]
    (ROOT/'FINAL_REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    concise=[
        '# Mental-Pong closeout summary',
        '**CLOSED_EXPLORATORY_WITH_LIMITATIONS**. A and1000 B/C repetitions are complete; each random repetition covers the original100 condition splits.',
        table(full,owncols),
        '1. **Task-position structure is readable.** Correct correspondence outperforms fixed-head mismatch. Actual candidates outperform1000 random-endpoint refits on full, hidden, and endpoint-influence r/RMSE.',
        '2. **No full-epoch candidate advantage.** Objective own-target reconstruction is better in all four groups. No-collision candidates have lower RMSE and lower correlation. Post-collision candidate RMSE has no consistent advantage over random candidates.',
        '3. **The full-epoch mean-endpoint geometry baseline remains stronger.** All four full-epoch skills are negative; some Perle hidden/no-collision configurations have slightly positive skill. Beating random targets does not establish general performance beyond shared geometry.',
        '4. **Behavioral cancellation is observed, but does not fully explain results.** Opposing signs occur in77/78 Mahler and78/78 Perle conditions, with median cancellation54.0%/48.0%. The trial-label comparison also changes condition weighting.',
        '**raw_preprocessing=fail.** Filling across conditions and times remains unresolved. Strictly pre-feedback endpoint timing and neural mean members are unverified. Within-condition opposite-behavior neural differences remain untested, not disproved. The [future design](future_design.md) was not executed.',
        '[Full report](FINAL_REPORT.md) · [Scientific acceptance](results/final_acceptance_audit.json) · [Verification and replay routes](README.md)'
    ]
    (ROOT/'FINAL_SUMMARY.md').write_text('\n\n'.join(concise)+'\n',encoding='utf-8')


if __name__=='__main__':run()
