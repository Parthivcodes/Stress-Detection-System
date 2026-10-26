# Active Demo Model Files

Generated-data simulation selected by models/training_preview.json. This index describes active files; it does not copy or rename hashed checkpoints.

| Algorithm | Checkpoint(s) | Encoder(s) |
|---|---|---|
| Logistic Regression | model_training/base_models/models/0_20261026.joblib | model_training/base_models/models/0_20261026.preprocessor.joblib |
| Random Forest | model_training/improved_models/models/Random_Forest_20261026.joblib | model_training/improved_models/models/Random_Forest_20261026.encoder.joblib |
| XGBoost | model_training/base_models/models/2_20261026.joblib | model_training/base_models/models/2_20261026.preprocessor.joblib |
| 1D CNN | model_training/improved_models/models/1D_CNN_20261026.pt<br>model_training/improved_models/models/1D_CNN_20261027.pt | model_training/improved_models/models/1D_CNN_20261026.encoder.joblib<br>model_training/improved_models/models/1D_CNN_20261027.encoder.joblib |
| LSTM | model_training/base_models/models/4_20261026.pt<br>model_training/base_models/models/4_20261027.pt | model_training/base_models/models/4_20261026.preprocessor.joblib<br>model_training/base_models/models/4_20261027.preprocessor.joblib |
| CNN-LSTM | model_training/base_models/models/5_20261026.pt<br>model_training/base_models/models/5_20261027.pt | model_training/base_models/models/5_20261026.preprocessor.joblib<br>model_training/base_models/models/5_20261027.preprocessor.joblib |

0=Logistic Regression,1=Random Forest,2=XGBoost,3=CNN,4=LSTM,5=CNN-LSTM. Seed numbers identify actual training runs. The real-data bundle is separately kept in models/current_models/.
