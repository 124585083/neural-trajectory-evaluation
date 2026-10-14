"""Render current Sensorium documentation figures from retained public records.

No model loading, inference, fitting, resampling, or statistical estimation occurs.
Stored points, contrasts, interval endpoints, and validity states are plotted directly.
The output manifest hashes every input and records the fields used by each figure.
Requires NumPy, pandas and Matplotlib; run with --help for the output contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/sensorium/evidence/04_model_comparison"
DEFAULT_OUT = ROOT / "results/sensorium/figures/documentation"
BLUE, ORANGE, TEAL, GRAY = "#27679A", "#C66B21", "#187A70", "#66717D"
INK, LIGHT = "#24313C", "#EEF3F6"
METRICS = ["position_correlation", "velocity_direction_cosine",
           "normalized_position_rmse", "speed_profile_correlation",
           "acceleration_direction_cosine"]
LABELS = ["Position r (primary)", "Direction cosine (primary)",
          "Position nRMSE ↓", "Speed r", "Acceleration cosine"]
SOURCES: dict[str, str] = {}
FIGURES: dict[str, dict] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(rel: str) -> Path:
    path = EVIDENCE / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    SOURCES[path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(record(rel))


def js(rel: str) -> dict:
    return json.loads(record(rel).read_text(encoding="utf-8"))


def one(df: pd.DataFrame, **where) -> pd.Series:
    found = df
    for col, value in where.items():
        found = found.loc[found[col] == value]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {where}; found {len(found)}")
    return found.iloc[0]


def require(condition, message: str):
    if not bool(condition):
        raise ValueError(message)


def tidy(ax, title: str, xlabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDE4E9", lw=.7, zorder=0)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, fc=LIGHT, color=INK, size=11):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.009,rounding_size=0.012",
                              facecolor=fc, edgecolor="#CCD6DE", linewidth=1))
    ax.text(x, y, text, ha="center", va="center", color=color,
            fontsize=size, linespacing=1.45)


def arrow(ax, a, b, **kwargs):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4, **kwargs))


def save(fig, output: Path, name: str, fields: dict, note: str):
    fig.canvas.draw()
    for ext in ("png", "svg"):
        metadata = {"Creator": "make_documentation_figures.py"}
        if ext == "svg":
            metadata["Date"] = None
        fig.savefig(output / f"{name}.{ext}", dpi=170, facecolor="white",
                    metadata=metadata)
    plt.close(fig)
    FIGURES[name] = {"source_fields": fields, "interpretation": note,
                     "outputs": {ext: sha(output/f"{name}.{ext}") for ext in ("png", "svg")}}


def shared_path(output: Path):
    cfg = js("retrained_v1/configuration.json")
    splits = csv("retrained_v1/split_summary.csv")
    gpfa = js("retrained_v1/frozen_gpfa_identity.json")
    summary = js("retrained_v1/evaluation_summary.json")
    counts = js("retrained_v1_diagnostics/encoder/parameter_counts.json")
    support = js("retrained_v1_diagnostics/encoder/temporal_support.json")
    response = csv("retrained_v1/response/response_session_metrics.csv")
    require(cfg["protocol"] == "retrained_v1", "Current protocol required")
    require(gpfa["fit_role"] == "train" and not gpfa["validation_in_final_fit"], "GPFA fit roles changed")
    require(not cfg["training"]["test_during_training"], "Test is not a training input")
    totals = splits.groupby("role")["n_trials"].sum().to_dict()
    n_neurons = response[(response.model == "static") &
                         (response.metric == "response_neuron_mean_r")].n_neurons.sum()
    require(n_neurons > 0, "Response population unavailable")
    dims = gpfa["latent_dim"]
    frames = cfg["data"]["frame_stop"] - cfg["data"]["frame_start"]
    obs = len(range(0, frames, cfg["gpfa"]["effective_observation_step"]))
    pilot = summary["pilot"]
    pilot_train_trials = int(one(splits, session=pilot["session"], role="train").n_trials)
    history = support["1.0"]["visual_and_appended_behavior_frame_offsets"]
    crop = -min(history)

    fig = plt.figure(figsize=(13.2, 10.2))
    ax = fig.add_axes([.025, .11, .95, .84]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    fig.suptitle('Sensorium: data roles and shared evaluation', y=.985, fontsize=17, weight="bold")
    roles = [("Train", "train", 'Encoder weights + preprocessing\nPilot training responses fit GPFA'),
             ("Validation", "validation", "Checkpoints + GPFA initialisation\nControl strengths"),
             ("Test", "test", 'Frozen scores, repeats + nulls\nRepeated test movies; earlier exploratory use')]
    for x, (title, role, detail) in zip([.165, .5, .835], roles):
        box(ax, x, .9, .30, .145, f"{title} · {totals[role]:,} trials\n{detail}", size=11)
    ax.text(.5, .795, 'Movie groups keep the same role across sessions; training fits models and validation selects settings.', ha="center", fontsize=10.5)
    box(ax, .14, .615, .23, .20, "Movie + 2 behaviour traces\n3 input channels\n\nPupil: separate pathway", size=11)
    box(ax, .465, .70, .30, .11,
        f"Static · frame t only\n{counts['counts']['static']['total']:,} parameters", fc="#E8F0F7")
    box(ax, .465, .525, .30, .13,
        f"Reduced Dynamic · t−{crop} … t\nVideo + behaviour history\n{counts['counts']['dynamic']['total']:,} parameters", fc="#FFF1E4")
    box(ax, .83, .615, .27, .20, "Separate learned readouts\n+ current-time pupil shifters\n\nPredictions for recorded neurons", size=11)
    arrow(ax, (.265,.67), (.305,.70)); arrow(ax, (.265,.56), (.305,.525))
    arrow(ax, (.625,.70), (.685,.665)); arrow(ax, (.625,.525), (.685,.56))
    box(ax, .5, .385, .93, .09, "Recorded responses + both prediction arrays → identical trial, neuron and scoring-time support", size=11.5)
    arrow(ax, (.83,.505), (.83,.44))
    box(ax, .245, .205, .435, .185,
        f"Response prediction\n{summary['response']['n_sessions']} sessions · {int(n_neurons):,} neurons\n\nPilot time-aware RSA / CKA\nDirect comparison of aligned responses", size=11)
    box(ax, .755, .205, .435, .185,
        f"Pilot trajectories · 1 session · {pilot_train_trials} training trials\n{pilot['n_test_conditions']} test movies · {cfg['data']['neurons']} neurons\nFixed scale → repeat means → same GPFA\n{obs} observed time points → {frames} inferred states\n× {dims} latent dimensions", size=10.5)
    arrow(ax, (.245,.332), (.245,.306)); arrow(ax, (.755,.332), (.755,.306))
    ax.text(.755,.062,'Encoder scale: time × neuron; GPFA scale: neuron.\nOffline full-window posterior in shared fixed coordinates.',ha="center",va="center",fontsize=10)
    ax.text(.245,.062,"Processed calcium responses; held-out after retraining.\nHistorical full Dynamic is a separate run.",ha="center",va="center",fontsize=10)
    timeline = fig.add_axes([.15,.035,.76,.062])
    for y,start,stop,label,color in [(2,0,299,"Input: 0–299",GRAY),(1,crop,299,f"Aligned: {crop}–299",BLUE),(0,cfg['data']['frame_start'],299,"Scored: 50–299",TEAL)]:
        timeline.plot([start,stop],[y,y],color=color,lw=6,solid_capstyle="butt")
        timeline.text(-10,y,label,ha="right",va="center",fontsize=10)
    timeline.set(xlim=(-5,305),ylim=(-.7,2.7),xticks=[0,crop,50,299],yticks=[])
    timeline.spines[:].set_visible(False)
    timeline.tick_params(axis="x",labelsize=9,length=0)
    save(fig,output,"01_shared_pathway",{"configuration.json":"data, gpfa, training, reporting",
         "split_summary.csv":"role / n_trials (sum across sessions)",
         "response_session_metrics.csv":"static response_neuron_mean_r / n_neurons",
         "parameter_counts.json":"counts", "temporal_support.json":"1.0 offsets",
         "frozen_gpfa_identity.json":"fit_role, latent_dim", "evaluation_summary.json":"pilot, response.n_sessions"},
         "Method schematic, not data points; original 30 Hz grid is not independent neural acquisition.")


def measurement(output: Path):
    repeats = csv("retrained_v1_diagnostics/gpfa/repeat_averaging_summary.csv")
    recon = csv("retrained_v1_diagnostics/gpfa/main/reconstruction.csv")
    levels = ["1", "2", "balanced"]
    actual = repeats.repeat_level.astype(str).unique().tolist()
    levels[-1] = next(x for x in actual if x not in ["1", "2"])
    fig, axes = plt.subplots(1,3,figsize=(13.2,4.8),gridspec_kw={"width_ratios":[1,1,1.1]})
    for ax,metric,label in zip(axes[:2],METRICS[:2],['A  Latent-position repeat agreement','B  Local-direction repeat agreement']):
        rows=[one(repeats,metric=metric,null="condition_shuffle",repeat_level=level) for level in levels]
        means=np.array([r.observed_mean for r in rows]); low=np.array([r.split_interval_low for r in rows]); high=np.array([r.split_interval_high for r in rows])
        require(all(r.splits == 200 for r in rows),"Unexpected repeat partition count")
        ax.errorbar(range(3),means,yerr=[means-low,high-means],fmt="o-",color=BLUE,capsize=4,lw=1.8)
        for x,v in enumerate(means): ax.annotate(f"{v:.3f}",(x,high[x]),xytext=(0,7),textcoords="offset points",fontsize=10,ha="center")
        ax.set(xticks=range(3),xticklabels=["1","2","Balanced\n4 / 5"],ylim=(0,1),xlim=(-.3,2.65))
        tidy(ax,label,'Repeats in each disjoint group'); ax.grid(axis="y",color="#E3E9ED",lw=.6)
    ax=axes[2]
    rows=[one(recon,profile="main",level=level,grid="full_query") for level in ["single_trial","condition_average"]]
    vals=np.array([r.population_r2_evaluation_mean for r in rows])*100
    ax.bar([0,1],vals,color=[GRAY,TEAL],width=.55)
    for x,v in enumerate(vals): ax.text(x,v+.12,f"{v:.2f}%",ha="center",fontsize=12,weight="bold")
    ax.set(xticks=[0,1],xticklabels=["Single trial","Condition\naverage"],ylim=(0,3.5),ylabel="Centred population R² (%)")
    tidy(ax,'C  Response variance reconstructed'); ax.grid(axis="x",visible=False)
    fig.suptitle('Repeatability and response-variance coverage of the four-dimensional measurement',fontsize=15,weight="bold",y=.98)
    fig.text(.5,.035,'A–B: 2.5th–97.5th percentiles across 200 repeat assignments of the same six movies.\nC: frozen reconstruction on 250 query times; each response level uses its own variance denominator.',ha="center",fontsize=10.5)
    fig.subplots_adjust(left=.06,right=.98,bottom=.24,top=.82,wspace=.4)
    save(fig,output,"02_measurement",{"repeat_averaging_summary.csv":"condition_shuffle / observed_mean, split_interval_low, split_interval_high; all three repeat levels",
        "main/reconstruction.csv":"profile=main, grid=full_query / population_r2_evaluation_mean"},
        "Repeat agreement and reconstruction are different measures; no reliability-normalised score or universal noise ceiling.")


def main_comparison(output: Path):
    sessions=csv("retrained_v1/response/response_session_paired_bootstrap.csv")
    paired=csv("retrained_v1/pilot/paired_bootstrap.csv")
    conditions=csv("retrained_v1/pilot/condition_metrics.csv")
    p=paired.set_index("metric").loc[METRICS]
    require((p.interval_estimand == "pooled_condition_trajectory").all(),"Trajectory interval target changed")
    require((p.valid_bootstraps == 2000).all() and (p.undefined_bootstraps == 0).all(),"Undefined primary comparison draws")
    require(p.loc["speed_profile_correlation","ci_low"] < 0 < p.loc["speed_profile_correlation","ci_high"],"Speed uncertainty changed")
    fig=plt.figure(figsize=(14,10.1)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.05],width_ratios=[1,1.05,1.05],hspace=.58,wspace=.7)
    ax=fig.add_subplot(gs[0,0]); s=sessions[sessions.metric=="response_neuron_mean_r"].sort_values("session")
    ids=[v.split("-Video-")[0].removeprefix("dynamic") for v in s.session]
    for i,(_,r) in enumerate(s.iterrows()):
        ax.plot([r.pooled_static,r.pooled_dynamic],[i,i],color="#AEBBC4",lw=1.4)
    ax.scatter(s.pooled_static,range(len(s)),color=BLUE,label="Static",zorder=3)
    ax.scatter(s.pooled_dynamic,range(len(s)),color=ORANGE,label="Reduced Dynamic",marker="s",zorder=3)
    ax.set(yticks=range(len(s)),yticklabels=ids,xlim=(.135,.212)); ax.invert_yaxis()
    tidy(ax,"A  Response r · five sessions",'Response r pooled across\ntest trials and time')
    ax.legend(loc="lower left",bbox_to_anchor=(-.08,-.38),frameon=False,ncol=1,fontsize=10)
    ax=fig.add_subplot(gs[0,1]); y=np.arange(5)
    ax.scatter(p.static,y,color=BLUE,zorder=3);ax.scatter(p.dynamic,y,color=ORANGE,marker="s",zorder=3)
    for i,(_,r) in enumerate(p.iterrows()):ax.plot([r.static,r.dynamic],[i,i],color="#BBC6CE",lw=1.5)
    ax.set(yticks=y,yticklabels=LABELS,xlim=(0,1.08));ax.invert_yaxis()
    tidy(ax,"B  Pilot trajectory scores","Absolute score (nRMSE: lower is better)")
    ax=fig.add_subplot(gs[0,2])
    for i,(_,r) in enumerate(p.iterrows()):
        color=GRAY if r.name=="speed_profile_correlation" else TEAL
        ax.errorbar(r.dynamic_minus_static,i,xerr=[[r.dynamic_minus_static-r.ci_low],[r.ci_high-r.dynamic_minus_static]],fmt="o",color=color,capsize=4,lw=1.7)
    ax.axvline(0,color=GRAY,lw=1); ax.set(yticks=y,yticklabels=LABELS,xlim=(-.57,.63));ax.invert_yaxis()
    tidy(ax,'C  Dynamic − Static\ntrajectory differences','Dynamic − Static; 95% paired\nmovie-bootstrap intervals')
    ax.text(.5,-.29,"Speed interval crosses zero",transform=ax.transAxes,color=GRAY,fontsize=10,ha="center")
    sub=gs[1,:].subgridspec(1,2,wspace=.5)
    movie_ids=sorted(conditions.condition.unique()); require(len(movie_ids)==6,"All six movies required")
    for j,metric in enumerate(METRICS[:2]):
        ax=fig.add_subplot(sub[0,j]); x=[];z=[]
        for movie in movie_ids:
            x.append(one(conditions,condition=movie,model="static",metric=metric).value)
            z.append(one(conditions,condition=movie,model="dynamic",metric=metric).value)
        for i,(a,b) in enumerate(zip(x,z)): ax.plot([a,b],[i,i],color="#B7C2CB",lw=1.6)
        ax.scatter(x,range(6),color=BLUE,s=40,zorder=3);ax.scatter(z,range(6),color=ORANGE,marker="s",s=40,zorder=3)
        ax.set(yticks=range(6),yticklabels=movie_ids);ax.invert_yaxis()
        tidy(ax, ["D  Position r · each test movie","E  Direction cosine · each test movie"][j],"Score against the same neural reference")
        if j==0:ax.set_xlim(-.52,.94)
        else:
            ax.set_xlim(-.04,.78)
            delta=z[0]-x[0];require(delta<0,"Preserve movie_00005 direction exception")
            ax.annotate(f"movie_00005: Δ = {delta:+.6f}", xy=(z[0], 0), xytext=(.68, .88),
                        textcoords="axes fraction", ha="center", fontsize=10, color=ORANGE,
                        arrowprops={"arrowstyle":"-", "color":ORANGE, "lw":.8})
    fig.suptitle('Static and reduced Dynamic: response and trajectory comparisons',fontsize=16,weight="bold",y=.975)
    fig.text(.5,.025,'A: all recorded neurons in five sessions. B–E: one pilot, 512 neurons, six movies.\nC: pooled trajectory scores; 2,000 saved paired whole-movie draws.\nEach row reports a separate metric; lower nRMSE indicates better agreement.',ha="center",fontsize=10.5)
    fig.subplots_adjust(left=.10,right=.97,bottom=.14,top=.89)
    save(fig,output,"03_model_comparison",{"response_session_paired_bootstrap.csv":"response_neuron_mean_r / pooled_static, pooled_dynamic (no interval attached)",
        "pilot/paired_bootstrap.csv":"five trajectory metrics / static, dynamic, dynamic_minus_static, ci_low, ci_high, interval_estimand",
        "pilot/condition_metrics.csv":"all six actual condition IDs / static,dynamic position and direction scores"},
        "Movie points are summary scores, not latent paths. Per-movie exception annotation is subtraction of two saved scores, not new inference.")


def reversal(output: Path):
    scores=csv("retrained_v1_diagnostics/reversal/metrics.csv")
    paired=csv("retrained_v1_diagnostics/reversal/paired_intervals.csv")
    checks=csv("retrained_v1_diagnostics/reversal/numerical_checks.csv")
    require(checks.passed.all(),"Reversal numerical checks must pass")
    pats=["cka_time_averaged_condition_pattern","rsa_time_averaged_condition_pattern"]
    for model in ["static","dynamic"]:
        for m in pats:
            require(abs(one(scores,model=model,metric=m,operation="original").value-one(scores,model=model,metric=m,operation="reversed").value)<1e-10,"Pattern invariance not retained")
    fig=plt.figure(figsize=(13.5,10.0));gs=fig.add_gridspec(3,2,height_ratios=[.8,1,1],hspace=.68,wspace=.4)
    ax=fig.add_subplot(gs[0,:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,.16,.61,.27,.57,"A → B → C\nC → B → A\nSame time average",fc="#EEF6F4",size=12)
    box(ax,.64,.72,.59,.35,'This diagnostic: prediction → reversal\n→ scale conversion → frozen GPFA',size=12)
    box(ax,.64,.22,.59,.35,'Primary null: prediction → scale conversion\n→ reversal → frozen GPFA',size=12)
    ax.text(.16,.12,"Encoder-scale outputs;\nneural reference stays fixed",ha="center",fontsize=10.5)
    ax.text(.64,-.10,"G(RP) ≠ R(GP): the scale depends on time",ha="center",va="center",fontsize=11,color=ORANGE)
    ax=fig.add_subplot(gs[1,0]); labs=[]
    for i,(model,m) in enumerate([(mo,me) for mo in ["static","dynamic"] for me in pats]):
        a=one(scores,model=model,metric=m,operation="original").value;b=one(scores,model=model,metric=m,operation="reversed").value
        ax.scatter(a,i,s=80,facecolors="none",edgecolors=BLUE,lw=1.8,label="Original" if i==0 else None)
        ax.scatter(b,i,s=30,color=ORANGE,marker="x",label="Reversed" if i==0 else None)
        labs.append(f"{'Static' if model=='static' else 'Dynamic'} · {'CKA' if m.startswith('cka') else 'RSA'}")
    ax.set(yticks=range(4),yticklabels=labs,xlim=(-.05,1));ax.invert_yaxis()
    tidy(ax,"A  Time-averaged patterns are unchanged","Score on six movie-average patterns")
    ax.legend(loc="lower right",frameon=False,fontsize=10)
    ax=fig.add_subplot(gs[1,1]); labs=[]
    for i,(model,m) in enumerate([(mo,me) for mo in ["static","dynamic"] for me in METRICS[:2]]):
        r=one(paired,model=model,metric=m)
        require(r.estimand=="pooled_condition_trajectory","Reversal trajectory target mismatch")
        ax.errorbar(r.b_minus_a,i,xerr=[[r.b_minus_a-r.ci_low],[r.ci_high-r.b_minus_a]],fmt="o",color=BLUE if model=="static" else ORANGE,capsize=4)
        labs.append(f"{'Static' if model=='static' else 'Dynamic'} · {'position' if m==METRICS[0] else 'direction'}")
    ax.axvline(0,color=GRAY,lw=1);ax.set(yticks=range(4),yticklabels=labs,xlim=(-.83,.05));ax.invert_yaxis()
    tidy(ax,"B  Primary trajectory agreement falls","Reversed − original · stored 95% intervals")
    conventional=["response_neuron_mean_r","cka_condition_time_aligned","cka_temporal_difference","rsa_condition_time_state_spearman"]
    clabels=["Response r","Time-aware CKA","Temporal-difference CKA","Time-state RSA"]
    for j,model in enumerate(["static","dynamic"]):
        ax=fig.add_subplot(gs[2,j])
        for i,m in enumerate(conventional):
            r=one(paired,model=model,metric=m)
            require(r.estimand=="equal_weight_mean_paired_condition_score","Reversal conventional target mismatch")
            color=GRAY if r.ci_low<=0<=r.ci_high else (BLUE if model=="static" else ORANGE)
            ax.errorbar(r.b_minus_a,i,xerr=[[r.b_minus_a-r.ci_low],[r.ci_high-r.b_minus_a]],fmt="o",color=color,capsize=4)
        ax.axvline(0,color=GRAY,lw=1);ax.set(yticks=range(4),yticklabels=clabels,xlim=(-.55,.065));ax.invert_yaxis()
        tidy(ax,f"{'C' if j==0 else 'D'}  {'Static' if model=='static' else 'Dynamic'} · time-aware summaries",'Equal-weight mean movie difference; 95% paired intervals')
    fig.suptitle('Prediction reversal: temporal sensitivity of response summaries',fontsize=16,weight="bold",y=.985)
    fig.text(.5,.025,'A: invariance verified by a deterministic numerical check. B: pooled trajectory contrasts. C–D: equal-weight movie-score contrasts.\nResponse alignment also deteriorates; Static temporal-difference CKA remains unresolved. Six movies; offline GPFA.',ha="center",fontsize=10.5)
    fig.subplots_adjust(left=.17,right=.98,bottom=.14,top=.91)
    save(fig,output,"04_reversal",{"reversal/metrics.csv":"original/reversed time-averaged condition-pattern RSA, CKA",
        "reversal/paired_intervals.csv":"a, b, b_minus_a, ci_low, ci_high; pooled_condition_trajectory vs equal_weight_mean_paired_condition_score kept separate",
        "reversal/numerical_checks.csv":"passed; numerical invariance"},
        "Method schematic plus saved comparisons; before-conversion reversal is not the after-conversion primary null.")


def matching(output: Path):
    selection=csv("retrained_v1/response_matching_selection.csv")
    scores=csv("retrained_v1/pilot/response_matching_metrics.csv")
    paired=csv("retrained_v1/pilot/response_matching_bootstrap.csv")
    summary=js("retrained_v1/evaluation_summary.json")["pilot"]["response_matching"]
    chosen=one(selection,selected=True)
    require(len(selection)==101 and summary["test_matching_retained"] and not summary["test_retuned"],"Frozen matching contract changed")
    require(chosen.absolute_gap==selection.absolute_gap.min(),"Selected amplitude must minimise validation gap")
    require(np.isclose(chosen.sigma,summary["sigma"]),"Frozen amplitude mismatch")
    vals=[chosen.validation_perturbed_dynamic_response_r-chosen.validation_static_response_r,
          one(scores,model="perturbed_dynamic",metric="response_neuron_mean_r").value-one(scores,model="static",metric="response_neuron_mean_r").value]
    require(np.allclose(np.abs(vals),[summary["validation_absolute_gap"],summary["test_absolute_gap"]]),"Response gap sources disagree")
    p=paired.set_index("metric").loc[METRICS]
    require((p.valid_bootstraps==2000).all() and (p.undefined_bootstraps==0).all(),"Undefined matching intervals")
    require(p.loc["speed_profile_correlation","ci_low"]<0<p.loc["speed_profile_correlation","ci_high"],"Preserve speed uncertainty")
    fig=plt.figure(figsize=(13.2,7.3));gs=fig.add_gridspec(2,2,height_ratios=[.52,1],hspace=.4,wspace=.55)
    ax=fig.add_subplot(gs[0,:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    for x,text in [(.16,"Train predictions\nSet one noise scale per neuron"),(.50,f"Validation\nSelect σ = {chosen.sigma:.6f}"),(.84,"Test\nApply fixed noise + clipping")]:
        box(ax,x,.65,.29,.45,text,size=11.5)
    arrow(ax,(.31,.65),(.35,.65));arrow(ax,(.655,.65),(.69,.65))
    ax.text(.5,.13,"Trial/time response r checks the match; repeat means + the frozen GPFA supply the trajectory scores.",ha="center",fontsize=11)
    ax=fig.add_subplot(gs[1,0]);tol=summary["tolerance"]
    ax.axvspan(-tol,tol,color="#E4F1E9",label=f"Fixed ±{tol:g} tolerance")
    ax.axvline(0,color=GRAY,lw=1)
    ax.scatter(vals,[1,0],color=[BLUE,TEAL],s=70,zorder=3)
    for v,y in zip(vals,[1,0]):ax.annotate(f"{v:+.6f}",(v,y),xytext=(0,14),textcoords="offset points",ha="center",weight="bold",fontsize=11)
    ax.set(yticks=[1,0],yticklabels=["Validation","Test"],ylim=(-.5,1.6),xlim=(-.007,.007))
    tidy(ax,'A  Mean response-correlation gap within tolerance',"Perturbed Dynamic − Static response r")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.33),frameon=False,fontsize=10)
    ax=fig.add_subplot(gs[1,1])
    for i,(_,r) in enumerate(p.iterrows()):
        v=r.perturbed_dynamic_minus_static
        ax.errorbar(v,i,xerr=[[v-r.ci_low],[r.ci_high-v]],fmt="o",color=ORANGE if r.name=="speed_profile_correlation" else TEAL,capsize=4,lw=1.7)
    ax.axvline(0,color=GRAY,lw=1);ax.set(yticks=range(5),yticklabels=LABELS,xlim=(-.54,.65));ax.invert_yaxis()
    tidy(ax,'B  Trajectory differences after response matching','Perturbed Dynamic − Static; 95% paired\nmovie-bootstrap intervals')
    fig.suptitle('Trajectory comparisons after matching mean response correlation',fontsize=16,weight="bold",y=.975)
    fig.text(.5,.028,'The tolerance constrains mean response r; per-neuron scores, NMSE and time-aware RSA/CKA remain separate outcomes.\nTrajectory intervals resample six whole movies with noise fixed; the speed difference remains unresolved.',ha="center",fontsize=10.5)
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.88)
    save(fig,output,"05_response_matching",{"response_matching_selection.csv":"selected row / sigma, validation_static_response_r, validation_perturbed_dynamic_response_r, absolute_gap",
        "response_matching_metrics.csv":"static and perturbed_dynamic / response_neuron_mean_r",
        "response_matching_bootstrap.csv":"all five metrics / perturbed_dynamic_minus_static, ci_low, ci_high",
        "evaluation_summary.json":"pilot.response_matching / tolerance, gap, retained, retuned"},
        "Signed response gaps subtract saved scores; bands are fixed tolerances, not confidence intervals. No new calibration or bootstrap.")


def retained_weight_figure():
    matches=csv("retrained_v1_diagnostics/encoder/spatial_response_matches.csv")
    zero=one(matches,temporal_retention=0.0)
    half=one(matches,temporal_retention=.5)
    require(zero.test_matching_status=="not_evaluable" and pd.isna(zero.spatial_test_r),"Spatial-zero correction must be retained")
    require(half.test_matching_status=="outside_tolerance","Half-retention mismatch must be retained")
    for rho in [.75,.25]:require(one(matches,temporal_retention=rho).test_matching_status=="matched","Valid matched pair changed")
    original=ROOT/"results/sensorium/figures/04_model_comparison/retrained_v1_diagnostics/encoder/weight_attenuation.png"
    SOURCES[original.relative_to(ROOT).as_posix()]=sha(original)
    FIGURES["06_weight_attenuation_retained"]={"existing_figure":original.relative_to(ROOT).as_posix(),
        "sha256":sha(original),"interpretation":"Recorded weight-control scores and match status. Spatial zero is NA/not_evaluable; temporal 0.50 fails test matching."}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT,help="Directory for five PNG/SVG figures and their source manifest (default: results/sensorium/figures/documentation)")
    args=parser.parse_args();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelsize":10.5,"xtick.labelsize":10,"ytick.labelsize":10.5,
                         "svg.fonttype":"none","svg.hashsalt":"sensorium-documentation-v1","axes.titlecolor":INK,"text.color":INK})
    for render in [shared_path,measurement,main_comparison,reversal,matching]:render(out)
    retained_weight_figure()
    manifest={"schema":"sensorium_documentation_figures_v1","protocol":"retrained_v1","script":Path(__file__).relative_to(ROOT).as_posix(),
              "script_sha256":sha(Path(__file__)),"operations":"Rendering saved values and method schematics only; no fitting, inference, resampling, new statistics, or control selection.",
              "software":{"matplotlib":matplotlib.__version__,"numpy":np.__version__,"pandas":pd.__version__},"sources":SOURCES,"figures":FIGURES}
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered 5 PNG/SVG figures; retained corrected weight figure. Manifest: {out/'source_manifest.json'}")


if __name__=="__main__":main()
