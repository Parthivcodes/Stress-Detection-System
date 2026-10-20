"""
Data Preprocessing, Feature Engineering, and Multimodal Partitioning
for Stress Level Detection Using Multimodal Physiological Signals.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ['Low Stress', 'Moderate Stress', 'High Stress']

def map_stress_category(stress_score: int) -> int:
    """
    Map original 3-8 discrete stress score to 3 standardized clinical categories:
    - Low Stress (0): score 3-4
    - Moderate Stress (1): score 5-6
    - High Stress (2): score 7-8
    """
    if stress_score <= 4:
        return 0  # Low Stress
    elif stress_score <= 6:
        return 1  # Moderate Stress
    else:
        return 2  # High Stress

def load_and_preprocess_data(csv_path: str = "data/original_sleep_dataset/sleep_lifestyle.csv"):
    """
    Loads raw CSV, cleans missing values, parses physiological signals (Blood Pressure),
    engineers domain-specific biomarkers, and encodes categorical attributes.
    """
    df = pd.read_csv(csv_path)

    # 1. Clean missing values and normalize text values
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

    # 2. Parse Multimodal Cardiovascular Signals: Blood Pressure (Systolic / Diastolic)
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = bp_split[0].astype(float)
    df['Diastolic_BP'] = bp_split[1].astype(float)

    # 3. Domain-Specific Physiological Biomarker Engineering
    # Pulse Pressure: Difference between systolic and diastolic BP (vascular elasticity)
    df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']
    
    # Mean Arterial Pressure (MAP): Average blood pressure in an individual during a cardiac cycle
    df['Mean_Arterial_Pressure'] = df['Diastolic_BP'] + (df['Pulse_Pressure'] / 3.0)
    
    # Rate-Pressure Product (RPP): Index of myocardial oxygen consumption / cardiovascular stress
    df['Rate_Pressure_Product'] = (df['Heart Rate'] * df['Systolic_BP']) / 100.0

    # Cardiovascular Stress Index: Combined strain from heart rate and systolic pressure
    df['Cardiovascular_Stress_Index'] = (df['Heart Rate'] / 72.0) * (df['Systolic_BP'] / 120.0)

    # Sleep Efficiency Index: Subjective sleep quality relative to hours slept
    df['Sleep_Efficiency'] = df['Quality of Sleep'] / (df['Sleep Duration'] + 0.1)

    # Heart Rate to Physical Activity Ratio: Physiological autonomic response
    df['HR_to_Activity_Ratio'] = df['Heart Rate'] / (df['Physical Activity Level'] + 1.0)

    # Hourly Awake Step Intensity: Activity density during waking hours
    awake_hours = (24.0 - df['Sleep Duration']).clip(lower=1.0)
    df['Steps_per_Hour_Awake'] = df['Daily Steps'] / awake_hours

    # 4. Target Labeling (3-class classification)
    df['Stress_Category'] = df['Stress Level'].apply(map_stress_category)

    # 5. Categorical Feature Encoding (One-Hot Encoding with clean columns)
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df_encoded

def get_modality_feature_groups(feature_columns):
    """
    Partitions engineered features into physiological modalities for ablation studies:
    - Cardiovascular (HR, BP, MAP, RPP, etc.)
    - Physical Activity & Movement (Daily Steps, Physical Activity Level, Awake intensity)
    - Sleep / Circadian (Sleep Duration, Sleep Quality, Sleep Disorders, Efficiency)
    - Demographics & Lifestyle (Age, Gender, Occupation, BMI)
    - Multimodal (All combined)
    """
    cardio_keywords = ['Heart Rate', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure', 
                       'Mean_Arterial_Pressure', 'Rate_Pressure_Product', 'Cardiovascular_Stress_Index', 'HR_to_Activity_Ratio']
    activity_keywords = ['Physical Activity Level', 'Daily Steps', 'Steps_per_Hour_Awake', 'HR_to_Activity_Ratio']
    sleep_keywords = ['Sleep Duration', 'Quality of Sleep', 'Sleep_Efficiency', 'Sleep Disorder']
    demographic_keywords = ['Age', 'Gender', 'Occupation', 'BMI Category']

    cardio_feats = [c for c in feature_columns if any(k in c for k in cardio_keywords)]
    activity_feats = [c for c in feature_columns if any(k in c for k in activity_keywords)]
    sleep_feats = [c for c in feature_columns if any(k in c for k in sleep_keywords)]
    demographic_feats = [c for c in feature_columns if any(k in c for k in demographic_keywords)]

    modalities = {
        'Cardiovascular': cardio_feats,
        'Physical_Activity': activity_feats,
        'Sleep_Circadian': sleep_feats,
        'Demographics': demographic_feats,
        'Multimodal_Physiological': list(set(cardio_feats + activity_feats + sleep_feats)),
        'Multimodal_Complete': list(feature_columns)
    }
    return modalities

def prepare_train_test_split(df_encoded, test_size: float = 0.20, val_size: float = 0.15, random_state: int = 42):
    """
    Performs stratified Train/Validation/Test split with zero data leakage.
    Scalers are strictly fitted only on the training set.
    """
    drop_cols = ['Person ID', 'Blood Pressure', 'Stress Level', 'Stress_Category']
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    
    X = df_encoded[feature_cols].values.astype(np.float32)
    y = df_encoded['Stress_Category'].values.astype(np.int64)

    # Step 1: Split into Train+Val and Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split Train+Val into Train and Validation
    val_ratio_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_train_val
    )

    # Step 3: Strict feature normalization (fit ONLY on X_train to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'modalities': get_modality_feature_groups(feature_cols)
    }

if __name__ == "__main__":
    df_clean = load_and_preprocess_data()
    print("Preprocessed shape:", df_clean.shape)
    splits = prepare_train_test_split(df_clean)
    print(f"Train samples: {len(splits['y_train'])}, Val samples: {len(splits['y_val'])}, Test samples: {len(splits['y_test'])}")
    print(f"Features total: {len(splits['feature_cols'])}")
    print("Class distribution in Train:", np.bincount(splits['y_train']))
