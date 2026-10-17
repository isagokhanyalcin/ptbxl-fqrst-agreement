"""Paper 1: inter-algorithm agreement of frontal QRS-T angle and low QRS voltage (PTB-XL+).
Usage: python3 -I paper1.py <data_dir> <out_dir>
"""
import sys, ast, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
import statsmodels.api as sm
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
D, O = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(2026)
R = {}  # key results for the write-up

# ---------------- data ----------------
db = pd.read_csv(f"{D}/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(f"{D}/scp_statements.csv", index_col=0)
db["scp"] = db.scp_codes.apply(ast.literal_eval)
diag = scp[scp.diagnostic == 1]
db["sup"] = db.scp.apply(lambda d: sorted({diag.loc[k, "diagnostic_class"] for k in d if k in diag.index}))
db["norm_only"] = db.sup.apply(lambda s: s == ["NORM"])
db["path"] = db.sup.apply(lambda s: len(s) > 0 and s != ["NORM"])
for c in ["MI", "STTC", "HYP", "CD"]:
    db[c] = db.sup.apply(lambda s, c=c: c in s)
db["lvolt_lbl"] = db.scp.apply(lambda d: "LVOLT" in d)
db["afib"] = db.scp.apply(lambda d: "AFIB" in d or "AFLT" in d)
db["bbb"] = db.scp.apply(lambda d: "CLBBB" in d or "CRBBB" in d)
db["pace"] = db.pacemaker.notna() | db.scp.apply(lambda d: "PACE" in d)
db["noise"] = db[["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]].notna().any(axis=1)
db["age_ok"] = db.age < 300
db["female"] = db.sex == 1

s12 = pd.read_csv(f"{D}/12sl_statements.csv", index_col="ecg_id")
db["lowv_12sl_stmt"] = s12.statements.apply(lambda x: "LOWV" in ast.literal_eval(x)).reindex(db.index)

limb = ["I", "II", "III", "aVR", "aVL", "aVF"]; prec = ["V1", "V2", "V3", "V4", "V5", "V6"]

def feats(alg, qcol):
    f = pd.read_csv(f"{D}/{alg}_features.csv").set_index("ecg_id")
    a = (f[qcol] - f.T_AxisFront_Global).abs()
    a = a.where(a <= 180, 360 - a)
    out = pd.DataFrame({"fq": a, "qrs_axis": f[qcol], "t_axis": f.T_AxisFront_Global,
                        "qrsd": f.QRS_Dur_Global, "hr": f.get("HR_Ventr_Global"), "qtc": f.get("QT_IntCorr_Global")})
    pl, pp = f[[f"QRS_AmpPP_{l}" for l in limb]].abs(), f[[f"QRS_AmpPP_{l}" for l in prec]].abs()
    out["lv_limb"] = (pl < 0.5).all(axis=1) & pl.notna().all(axis=1)
    out["lv_prec"] = (pp < 1.0).all(axis=1) & pp.notna().all(axis=1)
    out["lv_comb"] = out.lv_limb & out.lv_prec
    out["lv_any"] = out.lv_limb | out.lv_prec
    if "QRS_AxisIndet_Global" in f:
        out["indet"] = f.QRS_AxisIndet_Global == 1
    return out

ge = feats("12sl", "R_AxisFrontal_Global").add_suffix("_ge")
gl = feats("unig", "QRS_AxisFront_Global").add_suffix("_gl")
x = db.join(ge, how="inner").join(gl, how="inner")

# ---------------- cohort ----------------
flow = {"all_ecgs": len(db)}
x = x.sort_values(["patient_id", "recording_date"])
flow["with_both_feature_sets"] = len(x)
x = x[x.fq_ge.notna() & x.fq_gl.notna()]; flow["both_angles_available"] = len(x)
x = x[~x.pace]; flow["after_pacing_excluded"] = len(x)
x = x[~x.indet_gl.fillna(False)]; flow["after_indeterminate_axis_excluded"] = len(x)
allecg = x.copy()
x = x.groupby("patient_id", sort=False).head(1); flow["primary_first_ecg_per_patient"] = len(x)
R["flow"] = flow

# ---------------- helpers ----------------
def boot(fn, *arrs, B=1000):
    n = len(arrs[0]); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n); vals.append(fn(*[a[i] for a in arrs]))
    return np.percentile(vals, [2.5, 97.5])

def ccc(a, b):
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)

def ac1(a, b):
    a, b = np.asarray(a), np.asarray(b); cats = np.unique(np.r_[a, b]); q = len(cats)
    po = (a == b).mean(); pi = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    pe = (pi * (1 - pi)).sum() / (q - 1); return (po - pe) / (1 - pe)

def agree(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    return {"n": len(a), "pos_a": int(np.sum(a == True)) if a.dtype == bool else None,
            "obs_agree": po, "kappa": cohen_kappa_score(a, b), "pabak": 2 * po - 1, "ac1": ac1(a, b)}

def diag_acc(test, ref):
    test, ref = np.asarray(test, bool), np.asarray(ref, bool)
    tp, fn = (test & ref).sum(), (~test & ref).sum(); fp, tn = (test & ~ref).sum(), (~test & ~ref).sum()
    def wil(k, n):
        if n == 0: return (np.nan, np.nan, np.nan)
        p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (p, c - h, c + h)
    return {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn), "sens": wil(tp, tp + fn), "spec": wil(tn, tn + fp), "ppv": wil(tp, tp + fp)}

cat = lambda v: pd.cut(v, [-0.1, 45, 90, 180.1], labels=[0, 1, 2]).astype(int)

def agreement_block(d, label):
    a, b = d.fq_ge.values, d.fq_gl.values; diff = a - b; m = (a + b) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    out = {"label": label, "n": len(d), "median_ge": np.median(a), "iqr_ge": np.percentile(a, [25, 75]).tolist(),
           "median_gl": np.median(b), "iqr_gl": np.percentile(b, [25, 75]).tolist(),
           "bias": bias, "bias_ci": boot(lambda u, v: (u - v).mean(), a, b, B=500).tolist(), "sd": sd,
           "loa": [bias - 1.96 * sd, bias + 1.96 * sd], "abs_diff_median": np.median(np.abs(diff)),
           "pct_absdiff_gt30": 100 * (np.abs(diff) > 30).mean(), "spearman": stats.spearmanr(a, b)[0], "ccc": ccc(a, b)}
    ca, cb = cat(d.fq_ge), cat(d.fq_gl)
    out["wkappa3"] = cohen_kappa_score(ca, cb, weights="linear"); out["kappa3"] = cohen_kappa_score(ca, cb)
    out["reclass_pct"] = 100 * (ca != cb).mean()
    out.update({f"gt90_{k}": v for k, v in agree(d.fq_ge > 90, d.fq_gl > 90).items() if k != "pos_a"})
    out["gt90_pct_ge"], out["gt90_pct_gl"] = 100 * (d.fq_ge > 90).mean(), 100 * (d.fq_gl > 90).mean()
    return out

# ---------------- Table 1 ----------------
def t1(d):
    g = d[d.age_ok]
    return {"n": len(d), "age_median": g.age.median(), "age_iqr": g.age.quantile([.25, .75]).tolist(),
            "female_pct": 100 * d.female.mean(), "validated_pct": 100 * d.validated_by_human.mean(),
            "NORM_only_pct": 100 * d.norm_only.mean(), "MI_pct": 100 * d.MI.mean(), "STTC_pct": 100 * d.STTC.mean(),
            "HYP_pct": 100 * d.HYP.mean(), "CD_pct": 100 * d.CD.mean(), "afib_pct": 100 * d.afib.mean(),
            "bbb_pct": 100 * d.bbb.mean(), "noise_pct": 100 * d.noise.mean(),
            "qrsd_ge_median": d.qrsd_ge.median(), "hr_ge_median": d.hr_ge.median(), "qtc_ge_median": d.qtc_ge.median()}
T1 = pd.DataFrame({"All": t1(x), "NORM only": t1(x[x.norm_only]), "Any pathology": t1(x[x.path])}).T
T1.to_csv(f"{O}/table1_cohort.csv")

# ---------------- continuous + categorical agreement ----------------
blocks = [agreement_block(x, "Primary (first ECG per patient)")]
for lab, sub in [("QRS < 120 ms", x[x.qrsd_ge < 120]), ("QRS >= 120 ms", x[x.qrsd_ge >= 120]),
                 ("Sinus/no AF", x[~x.afib]), ("AF/flutter", x[x.afib]),
                 ("No noise flags", x[~x.noise]), ("Noise flagged", x[x.noise]),
                 ("Human-validated labels", x[x.validated_by_human]), ("All ECGs (incl. repeats)", allecg),
                 ("Limb low voltage (GE)", x[x.lv_limb_ge]), ("Normal voltage (GE)", x[~x.lv_limb_ge])]:
    if len(sub) > 30: blocks.append(agreement_block(sub, lab))
T2 = pd.DataFrame(blocks).set_index("label"); T2.to_csv(f"{O}/table2_fqrst_agreement.csv")

long = pd.concat([pd.DataFrame({"ecg": x.index, "alg": "GE", "v": x.fq_ge.values}),
                  pd.DataFrame({"ecg": x.index, "alg": "GL", "v": x.fq_gl.values})])
icc = pg.intraclass_corr(long, targets="ecg", raters="alg", ratings="v")
icc2 = icc.set_index("Type").loc["ICC(A,1)"]; R["icc_A1"] = {"icc": icc2.ICC, "ci": list(icc2["CI95"])}
# proportional bias
diff, mean = x.fq_ge - x.fq_gl, (x.fq_ge + x.fq_gl) / 2
pb = sm.OLS(diff, sm.add_constant(mean)).fit(); R["prop_bias_slope"] = {"b": pb.params.iloc[1], "p": pb.pvalues.iloc[1]}
R["crosstab3"] = pd.crosstab(cat(x.fq_ge), cat(x.fq_gl)).values.tolist()

# ---------------- low voltage ----------------
lv = []
for k in ["lv_limb", "lv_prec", "lv_comb", "lv_any"]:
    r = agree(x[f"{k}_ge"], x[f"{k}_gl"]); r.update({"def": k, "comparison": "GE vs Glasgow", "prev_ge": 100 * x[f"{k}_ge"].mean(), "prev_gl": 100 * x[f"{k}_gl"].mean()})
    lv.append(r)
    for alg in ["ge", "gl"]:
        r = agree(x[f"{k}_{alg}"], x.lvolt_lbl); r.update({"def": k, "comparison": f"{alg.upper()} vs cardiologist LVOLT"})
        r.update({f"{m}": v for m, v in diag_acc(x[f"{k}_{alg}"], x.lvolt_lbl).items()}); lv.append(r)
r = agree(x.lowv_12sl_stmt.fillna(False), x.lvolt_lbl); r.update({"def": "12SL LOWV statement", "comparison": "12SL statement vs cardiologist LVOLT"}); r.update(diag_acc(x.lowv_12sl_stmt.fillna(False), x.lvolt_lbl)); lv.append(r)
r = agree(x.lowv_12sl_stmt.fillna(False), x.lv_limb_ge); r.update({"def": "12SL LOWV statement", "comparison": "12SL statement vs GE limb measurement"}); lv.append(r)
T3 = pd.DataFrame(lv); T3.to_csv(f"{O}/table3_low_voltage.csv", index=False)
R["lvolt_label_n"] = int(x.lvolt_lbl.sum())
# superclass distribution of discordant low-voltage cases
R["lvolt_superclass"] = {g: {c: round(100 * d[c].mean(), 1) for c in ["MI", "STTC", "HYP", "CD"]} | {"NORM": round(100 * d.norm_only.mean(), 1), "n": len(d)}
                        for g, d in [("label+meas", x[x.lvolt_lbl & x.lv_limb_ge]), ("label only", x[x.lvolt_lbl & ~x.lv_any_ge]),
                                     ("meas only", x[~x.lvolt_lbl & x.lv_limb_ge])]}

# ---------------- determinants of discordance ----------------
m = x[x.age_ok].copy()
m["disc"] = ((m.fq_ge - m.fq_gl).abs() > 30).astype(int)
m["qrsd10"] = m.qrsd_ge / 10; m["age10"] = m.age / 10; m["hr10"] = m.hr_ge / 10
m["tamp_low"] = 0
preds = ["age10", "female", "qrsd10", "hr10", "bbb", "afib", "lv_limb_ge", "noise", "MI", "HYP"]
mm = m[preds + ["disc"]].dropna().astype(float)
lr = sm.Logit(mm.disc, sm.add_constant(mm[preds])).fit(disp=0)
T4 = pd.DataFrame({"OR": np.exp(lr.params), "lo": np.exp(lr.conf_int()[0]), "hi": np.exp(lr.conf_int()[1]), "p": lr.pvalues}).drop("const")
T4.to_csv(f"{O}/table4_discordance_predictors.csv"); R["disc_n_events"] = int(mm.disc.sum()); R["disc_n"] = len(mm)

# ---------------- diagnostic value (ROC) ----------------
def delong(y, s1, s2):
    y = np.asarray(y, bool); pos, neg = np.where(y)[0], np.where(~y)[0]
    def comps(s):
        X, Y = s[pos], s[neg]
        allv = np.r_[X, Y]; r = stats.rankdata(allv); rx = stats.rankdata(X); ry = stats.rankdata(Y)
        m_, n_ = len(X), len(Y); auc = (r[:m_].sum() - m_ * (m_ + 1) / 2) / (m_ * n_)
        v10 = (r[:m_] - rx) / n_; v01 = 1 - (r[m_:] - ry) / m_
        return auc, v10, v01
    a1, v10a, v01a = comps(np.asarray(s1, float)); a2, v10b, v01b = comps(np.asarray(s2, float))
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / len(pos) + S01 / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; z = (a1 - a2) / np.sqrt(var)
    se1, se2 = np.sqrt(S[0, 0]), np.sqrt(S[1, 1])
    return a1, (a1 - 1.96 * se1, a1 + 1.96 * se1), a2, (a2 - 1.96 * se2, a2 + 1.96 * se2), 2 * stats.norm.sf(abs(z))

roc_rows, curves = [], {}
for name, cases in [("Any pathology", x.path), ("MI", x.MI), ("STTC", x.STTC), ("HYP", x.HYP), ("CD", x.CD)]:
    sub = x[x.norm_only | cases]; y = cases.reindex(sub.index).values
    a1, c1, a2, c2, p = delong(y, sub.fq_ge.values, sub.fq_gl.values)
    fpr, tpr, thr = roc_curve(y, sub.fq_ge); j = np.argmax(tpr - fpr)
    roc_rows.append({"comparison": f"{name} vs NORM only", "n_cases": int(y.sum()), "n_norm": int((~y).sum()),
                     "auc_ge": a1, "ci_ge": c1, "auc_gl": a2, "ci_gl": c2, "p_delong": p,
                     "youden_thr_ge": thr[j], "sens_at_youden": tpr[j], "spec_at_youden": 1 - fpr[j],
                     "sens_at_90_ge": (sub.fq_ge[y] > 90).mean(), "spec_at_90_ge": (sub.fq_ge[~y] <= 90).mean()})
    curves[name] = {alg: roc_curve(y, sub[f"fq_{alg}"])[:2] for alg in ["ge", "gl"]}
T5 = pd.DataFrame(roc_rows); T5.to_csv(f"{O}/table5_roc.csv", index=False)

# ---------------- reference values ----------------
ref = x[x.norm_only & x.validated_by_human & x.age_ok & (x.age >= 18) & ~x.afib & ~x.noise].copy()
ref["agegrp"] = pd.cut(ref.age, [17, 29, 39, 49, 59, 69, 79, 120], labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"])
rows = []
for (g, sx), d in ref.groupby(["agegrp", "female"], observed=True):
    for alg in ["ge", "gl"]:
        v = d[f"fq_{alg}"]
        rows.append({"age": g, "sex": "F" if sx else "M", "alg": alg.upper(), "n": len(v),
                     "p2.5": v.quantile(.025), "median": v.median(), "p97.5": v.quantile(.975), "pct_gt90": 100 * (v > 90).mean()})
for alg in ["ge", "gl"]:
    for sx, d in ref.groupby("female"):
        v = d[f"fq_{alg}"]; rows.append({"age": "all", "sex": "F" if sx else "M", "alg": alg.upper(), "n": len(v), "p2.5": v.quantile(.025), "median": v.median(), "p97.5": v.quantile(.975), "pct_gt90": 100 * (v > 90).mean()})
T6 = pd.DataFrame(rows); T6.to_csv(f"{O}/table6_reference_values.csv", index=False)
R["ref_n"] = len(ref)
R["ref_age_trend_ge"] = stats.spearmanr(ref.age, ref.fq_ge)[0]

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (lab, sub) in zip(axs, [("QRS < 120 ms", x[x.qrsd_ge < 120]), ("QRS ≥ 120 ms", x[x.qrsd_ge >= 120])]):
    d_, m_ = sub.fq_ge - sub.fq_gl, (sub.fq_ge + sub.fq_gl) / 2
    ax.hexbin(m_, d_, gridsize=60, bins="log", cmap="Greys", mincnt=1, extent=(0, 180, -120, 120))
    b, s = d_.mean(), d_.std(ddof=1)
    for yv, ls in [(b, "-"), (b - 1.96 * s, "--"), (b + 1.96 * s, "--")]:
        ax.axhline(yv, color="#1f5fbf", ls=ls, lw=1)
    ax.text(178, b + 1.96 * s + 3, f"+1.96 SD {b + 1.96 * s:.1f}°", ha="right", color="#1f5fbf")
    ax.text(178, b - 1.96 * s - 9, f"−1.96 SD {b - 1.96 * s:.1f}°", ha="right", color="#1f5fbf")
    ax.text(178, b + 3, f"bias {b:.1f}°", ha="right", color="#1f5fbf")
    ax.set_title(f"{lab} (n = {len(sub):,})"); ax.set_xlabel("Mean of GE 12SL and Glasgow fQRS-T (°)"); ax.set_xlim(0, 180); ax.set_ylim(-120, 120)
axs[0].set_ylabel("Difference, GE 12SL − Glasgow (°)")
fig.tight_layout(); fig.savefig(f"{O}/fig1_bland_altman.png", dpi=300); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
for ax, (name, c) in zip(axs, curves.items()):
    for alg, col, lab in [("ge", "#1f5fbf", "GE 12SL"), ("gl", "#999999", "Glasgow")]:
        ax.plot(c[alg][0], c[alg][1], color=col, lw=1.2, label=lab)
    r = T5[T5.comparison.str.startswith(name)].iloc[0]
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8); ax.set_title(f"{name}\nAUC {r.auc_ge:.3f} vs {r.auc_gl:.3f}")
    ax.set_xlabel("1 − specificity")
axs[0].set_ylabel("Sensitivity"); axs[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{O}/fig2_roc.png", dpi=300); plt.close(fig)

# ---------------- dump ----------------
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, float): return round(o, 4)
    return o
json.dump(clean(R), open(f"{O}/key_results.json", "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
print(json.dumps(clean(R), ensure_ascii=False, indent=1))
print(T1.T); print(T2[["n", "bias", "loa", "pct_absdiff_gt30", "ccc", "wkappa3", "reclass_pct", "gt90_kappa", "gt90_ac1", "gt90_pct_ge", "gt90_pct_gl"]])
print(T3[["def", "comparison", "n", "obs_agree", "kappa", "ac1", "prev_ge", "prev_gl", "tp", "fn", "fp", "sens", "ppv"]] if "tp" in T3 else T3)
print(T4); print(T5); print(T6[T6.age == "all"]); print(T6[(T6.alg == "GE")])
