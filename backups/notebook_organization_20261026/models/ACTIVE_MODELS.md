# Active Demo Model Files

Generated-data simulation selected by models/training_preview.json. This index describes active files; it does not copy or rename hashed checkpoints.

| Algorithm | Checkpoint(s) | Encoder(s) |
|---|---|---|
| Logistic Regression | experiments/lower_noise_simulation_20261026/models/0_20261026.joblib | experiments/lower_noise_simulation_20261026/models/0_20261026.preprocessor.joblib |
| Random Forest | experiments/lower_noise_rf_cnn_tuning_20261026/models/Random_Forest_20261026.joblib | experiments/lower_noise_rf_cnn_tuning_20261026/models/Random_Forest_20261026.encoder.joblib |
| XGBoost | experiments/lower_noise_simulation_20261026/models/2_20261026.joblib | experiments/lower_noise_simulation_20261026/models/2_20261026.preprocessor.joblib |
| 1D CNN | experiments/lower_noise_rf_cnn_tuning_20261026/models/1D_CNN_20261026.pt<br>experiments/lower_noise_rf_cnn_tuning_20261026/models/1D_CNN_20261027.pt | experiments/lower_noise_rf_cnn_tuning_20261026/models/1D_CNN_20261026.encoder.joblib<br>experiments/lower_noise_rf_cnn_tuning_20261026/models/1D_CNN_20261027.encoder.joblib |
| LSTM | experiments/lower_noise_simulation_20261026/models/4_20261026.pt<br>experiments/lower_noise_simulation_20261026/models/4_20261027.pt | experiments/lower_noise_simulation_20261026/models/4_20261026.preprocessor.joblib<br>experiments/lower_noise_simulation_20261026/models/4_20261027.preprocessor.joblib |
| CNN-LSTM | experiments/lower_noise_simulation_20261026/models/5_20261026.pt<br>experiments/lower_noise_simulation_20261026/models/5_20261027.pt | experiments/lower_noise_simulation_20261026/models/5_20261026.preprocessor.joblib<br>experiments/lower_noise_simulation_20261026/models/5_20261027.preprocessor.joblib |

0=Logistic Regression,1=Random Forest,2=XGBoost,3=CNN,4=LSTM,5=CNN-LSTM. Seed numbers identify actual training runs. The real-data bundle is separately kept in models/current_models/.
