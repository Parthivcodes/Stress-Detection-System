# Synthetic stress simulation - educational experiment

**All 10,000 profiles and all labels are invented. No real BRFSS respondents were copied, resampled or modified. These are not patient records and these scores do not establish accuracy on real people.**

## What was built

The same 19 input names, valid ranges and category codes are retained. Age, sleep, physical unhealthy days, BMI and 15 categorical inputs are generated using a fixed random seed and hypothetical shared factors. The exact equation and its Gaussian uncertainty are in protocol.json and the generator. Coefficients are assumptions, not validated medical relationships. No clinical endorsement or calibrated probabilities are claimed.

There are 7,000 training-partition profiles, 1,500 validation profiles and 1,500 synthetic test profiles. Within training, 6,300 fit models and 700 select neural stopping epochs. Encoders are fitted only on the 6,300 fit profiles. All feature profiles and sample IDs are unique. No noise or label changes were made after looking at model results.

## Measured synthetic test results

Majority-class baseline: **37.73%** accuracy, **0%** high and sometimes recall. Mean raw-argmax accuracy across six models: **63.52%**.

| Model | Raw accuracy | Operating accuracy | Operating macro-F1 | Sometimes recall | High recall | High detected / missed |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 64.93% | 64.73% | 0.6437 | 45.51% | 75.00% | 333 / 111 |
| Random Forest | 61.00% | 60.93% | 0.6043 | 42.04% | 73.20% | 325 / 119 |
| XGBoost | 63.27% | 63.80% | 0.6354 | 46.12% | 73.42% | 326 / 118 |
| 1D CNN | 62.00% | 61.20% | 0.6150 | 49.39% | 70.50% | 313 / 131 |
| LSTM | 66.27% | 65.20% | 0.6505 | 48.78% | 75.23% | 334 / 110 |
| CNN-LSTM | 63.67% | 63.53% | 0.6340 | 46.53% | 75.68% | 336 / 108 |

The operating rule was selected on validation: protect raw validation sometimes recall, macro-F1 and QWK, and require high recall at least max(60%, raw high recall). If no feasible policy exists, raw argmax remains and the failed policy is disclosed. Validation constraints do not guarantee matching recalls on test. Full raw/operating precision, recall, F1, AUC, confusion matrices and individual-seed validation variability are saved.

These numbers cannot be compared with approximately69% on BRFSS as a before/after improvement: populations, label mechanisms and class proportions differ. The fact that this noisy simulation did not produce80% reinforces that generating rows does not guarantee a high score. An easier deterministic simulator can score much higher by construction without becoming more accurate on real people.

## Generation options

| Option | Suitable purpose | Limitation |
|---|---|---|
| Simple rule-based benchmark (toy_easy) | Classroom demonstration that all six algorithms can learn a known rule | High scores reflect how easy we made the invented rule |
| Noisy probabilistic simulation (simulation; completed) | Test ambiguity, minority classes and the complete ML pipeline | Unvalidated assumptions; no real-world accuracy claim |
| Harder overlapping simulation (stress_test) | Robustness and failure-case testing | Designed to be difficult, not to maximize accuracy |
| Real-distribution-fitted generator (not implemented) | Investigate training-only augmentation/privacy research | Requires real training data; assess utility on real held-out respondents. Cannot add absent predictive signal by assumption |

Recommendation: present the real-data model as the empirical study and the synthetic version as a separately labelled simulation. For stronger real-person performance, collect better distinguishing information and an independent external evaluation rather than changing labels to suit an accuracy target.

## Reproduce

Use the existing project Python environment with numpy, pandas, scikit-learn, XGBoost, PyTorch and joblib. No new ML packages were installed for this experiment. Choose a fresh output folder; the script refuses overwrite.

```powershell
python tools/synthetic_stress_experiment.py --project . --output experiments/synthetic_simulation_fresh --rows 10000 --scenario simulation
```

The other predeclared modes are toy_easy and stress_test. Changing mode changes the hypothetical label-generating assumptions and is a different simulation study. Do not choose or advertise a mode as an improvement on real stress prediction.

## Website PDF export

Download analysis PDF now saves a two-page readable report directly in the browser: summary, combined probability bars, all six individual predictions, all19 answers with category labels, date, and clear limitations. No new backend dependency is required. The build allow-list includes the exporter for future Vercel builds. Existing model artifacts and source datasets remain unchanged. The live online deployment has not been updated.

A real browser download was exercised on localhost; the resulting PDF was opened, text-checked and rendered on both pages. The saved sample uses the default illustrative UI profile, not a real individual. PDF generation uses no external service.

## Preservation and rollback

The synthetic models are stored only in this experiment. Nothing changes models/active_model.json or models/current_models/. Discarding this experiment is not necessary to restore predictions because it was never activated.

The original frontend files are backed up under backups/analysis_pdf_export_20261026. To revert the download UI, restore its frontend/prediction.js and frontend/index.html and tools/build_vercel.cjs into the corresponding project paths. The unused new frontend/analysis-pdf.js may be left in place. No model rollback is needed.

## Evidence and interpretation sources

- [US Census: What are synthetic data?](https://www.census.gov/about/what/synthetic-data.html)
- [UK Government: AI Insights - Synthetic Data](https://www.gov.uk/government/publications/ai-insights/ai-insights-synthetic-data-html)

The report and all numerical results are from this local experiment. External sources support the distinction between synthetic-data utility and real-world evaluation, not the invented coefficients.
