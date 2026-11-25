import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
O = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, TEAL, INK, MUTED, CARD, BG = "#13283f", "#0f766e", "#1b2430", "#4b5563", "#ffffff", "#eef2f6"
T2 = pd.read_csv(f"{O}/table2_fqrst_agreement.csv"); T6 = pd.read_csv(f"{O}/table6_reference_values.csv")
W, H = 3200, 1280
fig = plt.figure(figsize=(W/300, H/300), dpi=300); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, 230, color=NAVY))
ax.text(80, 105, "Same ECG, different angle", color="white", fontsize=22, weight="bold", va="center")
ax.text(84, 185, "Frontal QRS-T angle: GE 12SL versus Glasgow on the same recordings", color="#d6e2ee", fontsize=10.5, va="center")
ax.text(W-80, 85, "PTB-XL / PTB-XL+", color="#d6e2ee", fontsize=10.5, weight="bold", ha="right", va="center")
ax.text(W-80, 160, "18,570 ECGs", color="white", fontsize=18, weight="bold", ha="right", va="center")
def card(x, w, title, sub):
    ax.add_patch(FancyBboxPatch((x, 275), w, 715, boxstyle="round,pad=0,rounding_size=28", fc=CARD, ec="#dde3ea", lw=1.2))
    ax.text(x+45, 345, title, fontsize=13, weight="bold", color=INK, va="center")
    ax.text(x+45, 410, sub, fontsize=9, color=MUTED, va="center")
# card 1: agreement
card(80, 960, "Agreement on average", "Bland-Altman, 12SL minus Glasgow")
ax.text(125, 545, "−1.3°", fontsize=32, weight="bold", color=TEAL, va="center")
ax.text(560, 515, "mean difference", fontsize=10.5, color=INK, va="center")
ax.text(560, 580, "ICC 0.96", fontsize=12, weight="bold", color=INK, va="center")
ax.plot([125, 995], [680, 680], color="#dde3ea", lw=1)
ax.text(125, 760, "95% limits of agreement", fontsize=10.5, color=INK, weight="bold", va="center")
ax.text(125, 840, "−27.6° to +24.9°", fontsize=20, weight="bold", color=NAVY, va="center")
ax.text(125, 925, "AF/flutter: −49.9° to +43.1°", fontsize=10.5, color=MUTED, va="center")
# card 2: reclassified
card(1100, 960, "Different category", "Angle ≤45°, 45–90° or >90°, by program")
for y, v, lab, n, fc in [(470, "8.2%", "All ECGs", "1,518 / 18,570", "#e7f3f1"), (715, "15.0%", "AF or flutter", "in 1,172 ECGs", "#eef1f4")]:
    ax.add_patch(FancyBboxPatch((1145, y), 870, 215, boxstyle="round,pad=0,rounding_size=20", fc=fc, ec="none"))
    ax.text(1185, y+90, v, fontsize=28, weight="bold", color=TEAL if y < 600 else NAVY, va="center")
    ax.text(1185, y+170, n, fontsize=10, color=INK, va="center")
    ax.text(1700, y+105, lab, fontsize=11, weight="bold", color=INK, va="center")
# card 3: reference limits chart
card(2120, 1000, "Upper normal limit by age", "97.5th percentile, GE 12SL, 5,471 healthy adults")
cax = fig.add_axes([(2120+120)/W, 1-(940)/H, 820/W, 470/H]); cax.set_facecolor("white")
order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"]
for sx, col, lab, ls in [("M", NAVY, "Men", "-"), ("F", TEAL, "Women", "--")]:
    d = T6[(T6.alg == "GE") & (T6.sex == sx)].set_index("age").reindex(order)
    cax.plot(range(7), d["p97.5"], color=col, ls=ls, lw=2.4, marker="o", ms=5, label=lab)
cax.axhline(90, color="#c2410c", ls=":", lw=1.6); cax.text(0, 92, "Common 90° cut-off", color="#c2410c", fontsize=9.5)
cax.set_ylim(40, 100); cax.set_xticks(range(7), ["18–29", "30s", "40s", "50s", "60s", "70s", "≥80"], fontsize=9)
cax.tick_params(axis="y", labelsize=9); cax.set_ylabel("Degrees", fontsize=9.5)
for s in ["top", "right"]: cax.spines[s].set_visible(False)
cax.legend(frameon=False, fontsize=9.5, loc="lower right")
# bottom bar
ax.add_patch(FancyBboxPatch((80, 1035), W-160, 185, boxstyle="round,pad=0,rounding_size=28", fc=NAVY, ec="none"))
ax.text(130, 1127, "Low QRS voltage (limb leads)", color="white", fontsize=12.5, weight="bold", va="center")
ax.text(1100, 1127, "12SL vs Glasgow: κ 0.92", color="white", fontsize=12.5, weight="bold", va="center")
ax.text(1950, 1127, "Measured vs cardiologist label: κ 0.14", color="#d6e2ee", fontsize=12.5, weight="bold", va="center")
fig.savefig("Graphical_abstract.png", dpi=300, facecolor=BG)
from PIL import Image
im = Image.open("Graphical_abstract.png").convert("RGB"); print(im.size)
im.save("Graphical_abstract.tiff", compression="tiff_lzw", dpi=(300, 300))
