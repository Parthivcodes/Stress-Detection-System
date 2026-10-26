# Why the generated-data models are near65%

The simulator adds independent Gaussian noise with standard deviation **1.00** to its hidden score before assigning the three labels. That uncertainty is not an input feature. Identical input patterns can therefore produce different valid labels under repeated simulation.

Using the exact known generating equation, the expected best unconstrained classification accuracy over the7,000training feature profiles is **66.19%**. The approximate interval due to sampling input profiles is **65.79% to 66.58%**. This interval is not a confidence interval for a particular trained model's test score.

A predictor given the exact equation's conditional probabilities reaches **64.53%** on the existing1,500validation labels. This is a fixed analytic reference, not a trained candidate. No test labels or test performance were used in this calculation.

## Meaning for80%

Tuning can reduce a model's gap to the best predictor supported by these inputs, but cannot recover independent random uncertainty.80% is not a defensible expected target for this fixed simulator. Requiring preserved high and sometimes recall can only constrain the achievable operating accuracy further. Finite observed scores can fluctuate; the calculated value is not an absolute bound on every finite sample.

The earlier63.52% mean on the generated test set therefore does not imply that another large hyperparameter sweep will reach80%. Compare all model families on the same validation partition before attributing small differences to algorithms.

| Model | Existing validation raw accuracy |
|---|---:|
| Logistic Regression | 65.53% |
| Random Forest | 61.00% |
| XGBoost | 62.33% |
| 1D CNN | 60.40% |
| LSTM | 64.27% |
| CNN-LSTM | 62.73% |

## Recommended action

Retain the current data and models. A bounded same-data tuning exercise could still improve weaker models by small amounts; it must use training-only fitting and validation selection, with protected minority recalls and no promised80% target. The simulator's known equation can serve as an analytic diagnostic, but presenting it as independently learned clinical knowledge would be misleading.

An easier simulation would need different assumptions about randomness and label overlap. That is a separately labelled benchmark, not an accuracy improvement on this unchanged dataset or on real people. Do not alter current labels, cherry-pick easy rows or use sample IDs/random-seed reconstruction to create apparent gains.

This mathematical diagnosis relies on the known simulation mechanism. It cannot establish a maximum accuracy for real survey data. All original and training-preview model artifacts and both source datasets were verified unchanged.
