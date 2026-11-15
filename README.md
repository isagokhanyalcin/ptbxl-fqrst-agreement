# Frontal QRS-T angle and low QRS voltage in PTB-XL+

Analysis code for the manuscript *Same ECG, different angle: inter-algorithm agreement and age-specific reference limits of the frontal QRS-T angle in 18,570 ECGs from PTB-XL+* (in preparation).

The study compares the GE Marquette 12SL and University of Glasgow (Uni-G) programs on the same 12-lead ECGs. It looks at four things: agreement on the frontal QRS-T angle, agreement between measured low QRS voltage and the cardiologists' low-voltage statement, the diagnostic value of the angle, and age- and sex-specific reference limits in adults with a normal ECG.

## Data

Both datasets are open on PhysioNet under CC BY 4.0. They are not stored in this repository.

- PTB-XL v1.0.3: Wagner P, et al. Sci Data. 2020;7:154. https://doi.org/10.13026/kfzx-aw45
- PTB-XL+ v1.0.1: Strodthoff N, et al. Sci Data. 2023;10:279. https://doi.org/10.13026/g6h6-7g88

## Reproducing the analysis

```bash
pip install -r requirements.txt
bash scripts/00_download_data.sh          # about 220 MB into data/
python scripts/01_analysis_paper1.py data results
python scripts/02_make_manuscript_tables.py results results/manuscript_tables.md
```

The script writes these files to `results/`:

| File | Content |
|---|---|
| `table1_cohort.csv` | Cohort characteristics |
| `table2_fqrst_agreement.csv` | Agreement on the frontal QRS-T angle, overall and by subgroup |
| `table3_low_voltage.csv` | Low-voltage agreement between programs and against the cardiologist statement |
| `table4_discordance_predictors.csv` | Logistic regression for a between-program difference above 30° |
| `table5_roc.csv` | ROC analysis with DeLong comparison |
| `table6_reference_values.csv` | Age- and sex-specific percentiles in healthy adults |
| `fig1_bland_altman.png`, `fig2_roc.png`, `fig3_reference_limits.png` | Figures |
| `key_results.json` | Patient flow and summary statistics |

Bootstrap steps use a fixed seed (2026), so reruns give identical numbers.

## Authors

İsa Gökhan Yalçın, Ahmet Yunus Hatip, Furkan Çağrı Oğuzlar. Department of Emergency Medicine, Süleyman Demirel University, Isparta, Türkiye.

Contact: isayalcin@sdu.edu.tr
