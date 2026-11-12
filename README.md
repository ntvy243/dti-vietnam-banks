# Text-mined digital-transformation disclosure index (DTI) for Vietnamese banks

Code, dictionary and derived data for the MSIE 2027 paper (ID EA0004):
**"A Machine-Learning-Calibrated Text-Mining Instrument for Measuring Digital Transformation Disclosure: Evidence from Vietnamese Commercial Banks"**
Tuong Vy Nguyen and Yogi Tri Prasetyo, Yuan Ze University.

## Contents
| Path | What it is |
|---|---|
| `data/dictionary_120_terms.csv` | 120 Vietnamese keywords/phrases with their dimension (UX, ST, PL, OP, CU, DA). A string that appears under two dimensions is kept under the first. |
| `data/panel_dti_financials.csv` | Bank-year panel (278 rows, 26 banks, 2015-2025): DTI measures per dimension (S_DTI, LnDTI, W_DTI), token counts, ROA, ROE, TotalAssets, SIZE. **Raw merged file, before the cleaning steps below.** |
| `data/bank_performance_raw.csv` | Financial data as compiled by hand (ROE, ROA, total assets, ownership type). |
| `src/common.py` | Loading and the two samples used in the paper (see "Samples"). |
| `src/validation_designs.py` | Exclusion audit, ROA/ROE zero check, shuffled / bank-grouped / forward-in-time validation (Table 7). |
| `src/abl.py` | Ablation with bank-clustered bootstrap CIs, and SHAP in-sample vs held-out (Tables 5-6). |
| `src/reg.py`, `src/rob.py` | Panel regressions (Table 3) and robustness checks (Tables 8-9). |
| `src/hausman_check.py` | Hausman tests behind Table 3. |
| `src/exact_vs_fuzzy.py` | Exact vs fuzzy matching and generic-term test. Needs the report PDFs (not in this repo). |
| `src/make_coding_sample.py` | Builds the human-coding workbook. Needs the report PDFs. |
| `src/fig1.py`, `src/fig2.py` | Figures 1 and 2. |
| `coding/coding_sample_NCB_pilot.xlsx` | Pilot sheet for two human coders (precision / recall / kappa). |
| `results/` | Outputs of the scripts. |
| `figures/` | Vector (PDF/SVG) and 600-dpi PNG figures. |

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/validation_designs.py     # run all scripts from the repository root
python src/abl.py
python src/rob.py
python src/hausman_check.py
```
Scripts print tables and write CSVs to `results/`. `src/exact_vs_fuzzy.py` and `src/make_coding_sample.py` read PDFs from the folder in `REPORTS_DIR` (default `reports/`), e.g. `REPORTS_DIR=/path/to/pdfs python src/exact_vs_fuzzy.py`. Annual reports are public documents of the banks and are not redistributed here.

## Samples
* **Panel (N = 275):** raw file with `TotalAssets` entries stored as text in millions of VND divided by 1,000 (so SIZE = ln of total assets in trillions of VND) and BID 2016, 2019, 2022 removed (asset values do not belong to BID). ROA/ROE are winsorized at the 1st/99th percentile for the regressions. Built by `panel275()` in `src/common.py`.
* **ML sample as analysed (N = 248):** raw file minus the 30 rows whose `TotalAssets` is stored as text (`ml_sample()`); it still contains the three suspect BID rows, and ROA/ROE are not winsorized.
* **Unit-corrected and N = 245 samples** are sensitivity checks in `src/validation_designs.py`.
* State-owned banks: BID, CTG, VCB (`Ownership Type` in `bank_performance_raw.csv`).

## Known data issues (not fixed in the raw files)
* BID 2016/2019/2022 `TotalAssets` (about 76, 108, 129) look like other banks' values; BID 2023 ROA = 18.83 is implausible; BID 2025 and CTG 2024 repeat the previous year's assets.
* NCB (NVB) 2020-2022 and 2025 ROA/ROE round to 0.00: checked against NCB's annual reports, these are genuine near-zero profits, not missing values.
* Single-token dictionary entries ("ai", "lan", "mạng", ...) are ambiguous in ordinary Vietnamese; see the paper, Section 3.4.

## License and citation
Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0. Cite the paper once published (add DOI/Zenodo link here).

## How the DTI series is built (exact matching, 117 terms)
`python src/build_dti.py <folder_with_OCR_txt_reports> data/panel_dti_financials.csv` recomputes S_DTI, LnDTI = ln(1+F) and W_DTI (TF-IDF) for the six dimensions with **exact** diacritic-preserving phrase matching (word boundaries, flexible whitespace) and the 117-term dictionary. The three ambiguous single-token entries "ai", "lan" and "mạng" are flagged `in_main_index = False` in `data/dictionary_120_terms.csv`; "ai" is covered by "trí tuệ nhân tạo".

17 of the 278 OCR reports keep almost no Vietnamese diacritics (<10% accented letters) and are flagged `ocr_no_diacritics = 1`; they are excluded by `common.load()` (`results/ocr_excluded_reports.csv`). Re-running OCR on their PDFs would recover them. Resulting samples: 261 usable reports, panel N = 258, machine-learning sample N = 231, unit-corrected sample N = 257.

`data/panel_dti_financials_fuzzy_original.csv` keeps the earlier series produced with the accent-tolerant matcher of the original notebook; it is superseded and kept only for traceability.

## Reproduce
`python src/reg.py; python src/rob.py; python src/table4_models.py; python src/abl.py; python src/validation_designs.py; python src/fig2.py` (run from the repository root; outputs in `results/` and `figures/`).

## Dictionary validation (pilot, NCB only)
`coding/` holds the coded pilot: `precision_labels_NCB_pilot.csv` (186 matches, coder 1; coders 2 and 3 on a shared 50), `recall_labels_NCB_pilot.csv` (35 readable passages) and `validation_summary.csv` (precision 0.75, recall about 0.70, Cohen's kappa 0.91 between the two independent coders, Fleiss' kappa 0.76). Coder 1 reviewed preliminary candidate labels supplied with the sheet; coders 2 and 3 worked independently without them. Pilot only: one bank, drawn under the original 120-entry list.
