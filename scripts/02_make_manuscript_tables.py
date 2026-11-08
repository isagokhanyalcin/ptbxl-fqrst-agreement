"""Builds the manuscript tables (markdown) directly from the result CSVs.
Usage: python3 -I make_tables.py <results_dir> <out_md>"""
import sys, re, pandas as pd

O, OUT = sys.argv[1], sys.argv[2]


def nums(s):
    s = re.sub(r"np\.float64\(", "", str(s))
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s)]


def neg(x, d=1):
    """Format with a true minus sign."""
    return f"{x:.{d}f}".replace("-", "−")


f1 = lambda v: f"{v:.1f}"
out = []

# ---- Table 1
t1 = pd.read_csv(f"{O}/table1_cohort.csv", index_col=0)
cols = ["All", "NORM only", "Any pathology"]
row = lambda lbl, fn: "| " + lbl + " | " + " | ".join(fn(t1.loc[c]) for c in cols) + " |"
out += ["**Table 1.** Characteristics of the primary cohort (first eligible ECG per patient).", "",
        "| Characteristic | All | Normal ECG only | Any abnormality |", "|---|---|---|---|",
        row("n", lambda r: f"{int(r.n):,}"),
        row("Age, years, median (IQR)*", lambda r: f"{r.age_median:.0f} ({nums(r.age_iqr)[0]:.0f}–{nums(r.age_iqr)[1]:.0f})"),
        row("Women, %", lambda r: f1(r.female_pct)),
        row("Report validated by a second cardiologist, %", lambda r: f1(r.validated_pct)),
        row("Myocardial infarction (MI), %", lambda r: f1(r.MI_pct)),
        row("ST/T change (STTC), %", lambda r: f1(r.STTC_pct)),
        row("Hypertrophy (HYP), %", lambda r: f1(r.HYP_pct)),
        row("Conduction disturbance (CD), %", lambda r: f1(r.CD_pct)),
        row("Atrial fibrillation or flutter, %", lambda r: f1(r.afib_pct)),
        row("Complete bundle branch block, %", lambda r: f1(r.bbb_pct)),
        row("Any signal quality flag, %", lambda r: f1(r.noise_pct)),
        row("QRS duration, ms, median†", lambda r: f"{r.qrsd_ge_median:.0f}"),
        row("Heart rate, beats/min, median†", lambda r: f"{r.hr_ge_median:.0f}"),
        row("QTc, ms, median†", lambda r: f"{r.qtc_ge_median:.0f}"),
        "", "*Ages above 89 years are masked in PTB-XL and were excluded from age summaries. †12SL measurements. "
        "Superclasses are not mutually exclusive; 71 ECGs had no diagnostic superclass.", ""]

# ---- Table 2
t2 = pd.read_csv(f"{O}/table2_fqrst_agreement.csv", index_col=0)
names = {"Primary (first ECG per patient)": "Primary cohort", "QRS < 120 ms": "QRS < 120 ms", "QRS >= 120 ms": "QRS ≥ 120 ms",
         "Sinus/no AF": "No AF/flutter", "AF/flutter": "AF/flutter", "No noise flags": "No quality flag",
         "Noise flagged": "Quality flag present", "Human-validated labels": "Validated reports",
         "All ECGs (incl. repeats)": "All ECGs incl. repeats", "Limb low voltage (GE)": "Limb-lead low voltage*",
         "Normal voltage (GE)": "Normal limb voltage*"}
out += ["**Table 2.** Agreement between GE 12SL and Glasgow (Uni-G) on the frontal QRS-T angle.", "",
        "| Group | n | Bias, ° (95% CI) | 95% LoA, ° | Difference > 30°, % | CCC | Weighted κ, 3 categories | Reclassified, % | κ at 90° | AC1 at 90° |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in names.items():
    r = t2.loc[k]; ci = nums(r.bias_ci); lo = nums(r.loa)
    out.append(f"| {v} | {int(r.n):,} | {neg(r.bias)} ({neg(ci[0])} to {neg(ci[1])}) | {neg(lo[0])} to {neg(lo[1])} | "
               f"{r.pct_absdiff_gt30:.1f} | {r.ccc:.2f} | {r.wkappa3:.2f} | {r.reclass_pct:.1f} | {r.gt90_kappa:.2f} | {r.gt90_ac1:.2f} |")
out += ["", "Bias = mean of 12SL minus Uni-G. Difference > 30° = absolute between-program difference. Categories: ≤45°, >45° to ≤90°, >90°. "
        "LoA, limits of agreement; CCC, Lin's concordance correlation coefficient; AF, atrial fibrillation. *By 12SL.", ""]

# ---- Table 3
t3 = pd.read_csv(f"{O}/table3_low_voltage.csv")
dn = {"lv_limb": "Limb leads (all six < 0.5 mV)", "lv_prec": "Precordial leads (all six < 1.0 mV)",
      "lv_comb": "Limb and precordial", "lv_any": "Limb or precordial", "12SL LOWV statement": "12SL LOWV statement"}
cmp = {"GE vs Glasgow": "12SL vs Uni-G", "GE vs cardiologist LVOLT": "12SL vs LVOLT", "GL vs cardiologist LVOLT": "Uni-G vs LVOLT",
       "12SL statement vs cardiologist LVOLT": "Statement vs LVOLT", "12SL statement vs GE limb measurement": "Statement vs 12SL limb measurement"}
out += ["**Table 3.** Low QRS voltage: agreement between programs and with the cardiologists' LVOLT statement (n = 18,570; LVOLT in 162).", "",
        "| Definition | Comparison | Prevalence, % | κ | AC1 | Sensitivity, % (95% CI) | PPV, % (95% CI) |", "|---|---|---|---|---|---|---|"]
for _, r in t3.iterrows():
    prev = f"{r.prev_ge:.2f} / {r.prev_gl:.2f}" if pd.notna(r.prev_ge) else "–"
    if isinstance(r.get("sens"), str):
        s, pv = nums(r.sens), nums(r.ppv)
        sens = f"{100*s[0]:.1f} ({100*s[1]:.1f}–{100*s[2]:.1f})"; ppv = f"{100*pv[0]:.1f} ({100*pv[1]:.1f}–{100*pv[2]:.1f})"
    else:
        sens = ppv = "–"
    out.append(f"| {dn[r['def']]} | {cmp[r.comparison]} | {prev} | {r.kappa:.2f} | {r.ac1:.2f} | {sens} | {ppv} |")
out += ["", "Prevalence is shown as 12SL / Uni-G for program comparisons. Sensitivity and PPV use the LVOLT statement as reference. "
        "AC1, Gwet's first-order agreement coefficient; PPV, positive predictive value.", ""]

# ---- Table 4 (ROC)
t5 = pd.read_csv(f"{O}/table5_roc.csv")
out += ["**Table 4.** Diagnostic value of the frontal QRS-T angle against ECGs labelled only as normal (n = 8,286).", "",
        "| Comparison | Cases, n | AUC 12SL (95% CI) | AUC Uni-G (95% CI) | P (DeLong) | Youden cut-off, ° | Sens/spec at Youden, % | Sens/spec at 90°, % |",
        "|---|---|---|---|---|---|---|---|"]
for _, r in t5.iterrows():
    a, b, p = nums(r.ci_ge), nums(r.ci_gl), r.p_delong
    ps = "< 0.001" if p < 0.001 else f"{p:.3f}"
    out.append(f"| {r.comparison.replace(' vs NORM only', '')} | {int(r.n_cases):,} | {r.auc_ge:.3f} ({a[0]:.3f}–{a[1]:.3f}) | "
               f"{r.auc_gl:.3f} ({b[0]:.3f}–{b[1]:.3f}) | {ps} | {r.youden_thr_ge:.0f} | {100*r.sens_at_youden:.1f} / {100*r.spec_at_youden:.1f} | "
               f"{100*r.sens_at_90_ge:.1f} / {100*r.spec_at_90_ge:.1f} |")
out += ["", "Cut-offs, sensitivity (sens) and specificity (spec) refer to 12SL. Any pathology = at least one abnormal diagnostic superclass.", ""]

# ---- Table 5 (reference)
t6 = pd.read_csv(f"{O}/table6_reference_values.csv")
order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80", "all"]
out += ["**Table 5.** Frontal QRS-T angle (°) in healthy adults by age group and sex: median (2.5th–97.5th percentile).", "",
        "| Age, years | Sex | n | 12SL | Uni-G |", "|---|---|---|---|---|"]
for a in order:
    for sx in ["M", "F"]:
        g = t6[(t6.age == a) & (t6.sex == sx) & (t6.alg == "GE")].iloc[0]
        u = t6[(t6.age == a) & (t6.sex == sx) & (t6.alg == "GL")].iloc[0]
        mark = "*" if bool(g.provisional_n_lt_120) else ""
        al = {"all": "All ages", ">=80": "≥80"}.get(a, a.replace("-", "–"))
        out.append(f"| {al} | {'Men' if sx == 'M' else 'Women'} | {int(g.n):,}{mark} | {g['median']:.0f} ({g['p2.5']:.0f}–{g['p97.5']:.0f}) | "
                   f"{u['median']:.0f} ({u['p2.5']:.0f}–{u['p97.5']:.0f}) |")
out += ["", "Healthy adults: age ≥ 18 years, ECG labelled only as normal, report validated by a second cardiologist, no atrial fibrillation or flutter "
        "and no signal quality flag (n = 5,471). Nonparametric percentiles. *Below the CLSI EP28-A3c minimum of 120 per partition; provisional.", ""]

# ---- Supplementary S1
t4 = pd.read_csv(f"{O}/table4_discordance_predictors.csv", index_col=0)
lab = {"age10": "Age (per 10 years)", "female": "Female sex", "qrsd10": "QRS duration (per 10 ms)", "hr10": "Heart rate (per 10 beats/min)",
       "bbb": "Complete bundle branch block", "afib": "Atrial fibrillation or flutter", "lv_limb_ge": "Limb-lead low voltage",
       "noise": "Any signal quality flag", "MI": "Myocardial infarction", "HYP": "Hypertrophy"}
out += ["**Supplementary Table S1.** Predictors of a between-program difference in the frontal QRS-T angle above 30° "
        "(multivariable logistic regression; 593 events among 18,324 ECGs).", "", "| Variable | OR (95% CI) | P |", "|---|---|---|"]
for k, v in lab.items():
    r = t4.loc[k]; ps = "< 0.001" if r.p < 0.001 else f"{r.p:.3f}"
    out.append(f"| {v} | {r.OR:.2f} ({r.lo:.2f}–{r.hi:.2f}) | {ps} |")
out.append("")
open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
