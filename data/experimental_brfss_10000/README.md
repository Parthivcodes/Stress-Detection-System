# BRFSS 10,000-respondent experimental subset

Exactly 10,000 real respondents from the existing BRFSS 2022 dataset. The 19 predictors, labels, original codes and provenance columns are unchanged. This is a smaller subset of the existing data, not a newly collected or independently validated dataset.

## How records were selected

Proportional sampling within each original split and target class, using a single fixed SHA-256 ranking of respondent IDs. No model scores, correctness, confidence or desired accuracy were used. Previously quarantined records remain excluded. Different people with identical feature values were retained.

| Split | Rarely | Sometimes | High | Total |
|---|---:|---:|---:|---:|
| train | 4850 | 1646 | 868 | 7364 |
| validation | 573 | 197 | 102 | 872 |
| test | 1160 | 396 | 208 | 1764 |

## Safe use

Use only the 19 predictors listed in protocol.json. Use target as the label. Respondent IDs, source rows, split, profile_group, stress_raw and raw_* columns are provenance/label metadata, not additional model inputs. Feeding stress_raw or raw_stress_raw into a classifier would leak the answer.

Keep the split column. Do not randomly reshuffle existing validation/test people into training. Fit preprocessing only on train. Use profile_group for training-only grouped cross-validation. The validation/test rows belong to historically reused internal splits; they are not a fresh untouched test.

The existing models have already seen the selected training rows. Evaluating those models on the entire CSV would produce an invalid estimate of generalization. A genuine subset experiment must retrain models on the subset training rows and evaluate against a fixed comparison set.

## Quality checks and limitations

All checks passed: 10,000 unique IDs, exact original values, valid feature ranges/target mapping, no missing predictors and no cross-split profile-group overlap. 0 rows share all 19 features with another selected respondent. 0 such feature groups contain different labels; these records were not removed to make classification easier.

This subset preserves the quality limitations of the source survey. Reducing row count does not remove measurement uncertainty or create information absent from the predictors. No accuracy improvement or 80% average accuracy has been demonstrated. No models were retrained and no active data/model paths changed.

The original data and models remain intact. quality_report.json records checks and hashes; protocol.json records the selection method. The standalone selection script makes the extraction reproducible.
