"""
LSTM Customer Churn Prediction System
- Bidirectional LSTM with Attention
- Behavioral Drift Analysis
- Explainable AI (SHAP)
- Comprehensive metrics saved for dashboard
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Bidirectional,
                                     Input, Layer, Attention, Concatenate,
                                     Flatten, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============================================================================
#  1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess():
    print("=" * 60)
    print("  LOADING CHURN NOISY DATASET")
    print("=" * 60)

    # Load the named dataset (created by analyze_noisy.py)
    if os.path.exists('churn_noisy_named.xlsx'):
        df = pd.read_excel('churn_noisy_named.xlsx')
        print(f"  Loaded churn_noisy_named.xlsx: {df.shape}")
    else:
        # Fallback: load raw and assign names
        df = pd.read_excel('churn_noisy.xlsx', header=None)
        df.columns = [
            'AccountAge', 'DataUsage', 'Age', 'MonthlySpend',
            'CallDuration', 'CallFrequency', 'SMSFrequency', 'DataFrequency',
            'PlanDuration', 'OverageFee', 'RoamingCalls',
            'CustomerSatisfaction', 'ComplaintRate', 'Churn',
            'AgeGroup', 'PlanType', 'Status'
        ]
        print(f"  Loaded churn_noisy.xlsx (raw): {df.shape}")

    # --- Feature Engineering ---
    # Encode categorical columns
    le_age = LabelEncoder()
    le_plan = LabelEncoder()
    le_status = LabelEncoder()

    df['AgeGroup_enc'] = le_age.fit_transform(df['AgeGroup'])
    df['PlanType_enc'] = le_plan.fit_transform(df['PlanType'])
    df['Status_enc'] = le_status.fit_transform(df['Status'])

    # Interaction features for behavioral drift analysis
    df['Usage_Intensity'] = df['DataUsage'] * df['CallFrequency']
    df['Spend_Satisfaction'] = df['MonthlySpend'] * df['CustomerSatisfaction']
    df['Complaint_Usage_Ratio'] = df['ComplaintRate'] / (df['DataUsage'] + 0.01)
    df['Engagement_Score'] = (df['CallDuration'] + df['SMSFrequency'] + df['DataFrequency']) / 3
    df['Loyalty_Index'] = df['AccountAge'] * df['CustomerSatisfaction']
    df['Risk_Score'] = df['ComplaintRate'] * df['RoamingCalls'] / (df['CustomerSatisfaction'] + 0.01)

    # Target
    y = df['Churn'].astype(int)

    # Features - drop non-numeric and target
    feature_cols = [
        'AccountAge', 'DataUsage', 'Age', 'MonthlySpend',
        'CallDuration', 'CallFrequency', 'SMSFrequency', 'DataFrequency',
        'PlanDuration', 'OverageFee', 'RoamingCalls',
        'CustomerSatisfaction', 'ComplaintRate',
        'AgeGroup_enc', 'PlanType_enc', 'Status_enc',
        'Usage_Intensity', 'Spend_Satisfaction', 'Complaint_Usage_Ratio',
        'Engagement_Score', 'Loyalty_Index', 'Risk_Score'
    ]

    X = df[feature_cols].copy()
    feature_names = feature_cols

    # Handle any infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    print(f"  Features: {len(feature_names)}")
    print(f"  Churn ratio: {y.mean() * 100:.1f}%")

    # Save label encoders
    joblib.dump(le_age, 'le_age.pkl')
    joblib.dump(le_plan, 'le_plan.pkl')
    joblib.dump(le_status, 'le_status.pkl')

    return X_scaled, y, scaler, feature_names, df


# ============================================================================
#  2. BEHAVIORAL DRIFT ANALYSIS
# ============================================================================

def compute_drift_analysis(df, feature_names):
    """
    Simulate behavioral drift by splitting data into temporal segments
    and measuring statistical shifts in feature distributions.
    """
    print("\n" + "=" * 60)
    print("  BEHAVIORAL DRIFT ANALYSIS")
    print("=" * 60)

    # Split data into 5 temporal windows (simulating time periods)
    n = len(df)
    window_size = n // 5
    windows = []
    for i in range(5):
        start = i * window_size
        end = start + window_size if i < 4 else n
        windows.append(df.iloc[start:end])

    drift_results = {}
    numerical_features = [
        'AccountAge', 'DataUsage', 'MonthlySpend', 'CallDuration',
        'CallFrequency', 'SMSFrequency', 'DataFrequency',
        'CustomerSatisfaction', 'ComplaintRate', 'RoamingCalls'
    ]

    for feat in numerical_features:
        means = [w[feat].mean() for w in windows]
        stds = [w[feat].std() for w in windows]

        # Population Stability Index (PSI) between first and last window
        p = windows[0][feat].value_counts(bins=10, normalize=True).sort_index()
        q = windows[-1][feat].value_counts(bins=10, normalize=True).sort_index()
        # Align bins
        p, q = p.align(q, fill_value=0.0001)
        p = p.clip(lower=0.0001)
        q = q.clip(lower=0.0001)
        psi = np.sum((p - q) * np.log(p / q))

        # Mean drift (absolute change from first to last window)
        mean_drift = abs(means[-1] - means[0])
        std_drift = abs(stds[-1] - stds[0])

        drift_results[feat] = {
            'window_means': [round(m, 4) for m in means],
            'window_stds': [round(s, 4) for s in stds],
            'psi': round(float(psi), 4),
            'mean_drift': round(float(mean_drift), 4),
            'std_drift': round(float(std_drift), 4),
            'drift_severity': 'High' if psi > 0.2 else ('Medium' if psi > 0.1 else 'Low')
        }

        print(f"  {feat:<25} PSI={psi:.4f}  Drift={drift_results[feat]['drift_severity']}")

    # Churn rate drift across windows
    churn_rates = [w['Churn'].mean() * 100 for w in windows]
    drift_results['churn_rate_by_window'] = [round(cr, 2) for cr in churn_rates]
    print(f"\n  Churn rates by window: {[f'{cr:.1f}%' for cr in churn_rates]}")

    return drift_results


# ============================================================================
#  3. BUILD LSTM MODEL WITH ATTENTION
# ============================================================================

def build_lstm_model(n_features):
    """Bidirectional LSTM with Self-Attention mechanism."""

    inp = Input(shape=(1, n_features), name='input')

    # First BiLSTM
    x = Bidirectional(LSTM(256, return_sequences=True, name='lstm_1'),
                      name='bilstm_1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Second BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True, name='lstm_2'),
                      name='bilstm_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Third BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True, name='lstm_3'),
                      name='bilstm_3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Self-Attention
    attention = Attention(name='attention')([x, x])
    x = Concatenate()([x, attention])
    x = Flatten()(x)

    # Dense layers
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation='relu', name='dense_3')(x)

    out = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inp, outputs=out, name='BiLSTM_Attention_Churn')
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n  Model Architecture:")
    model.summary()
    return model


# ============================================================================
#  4. TRAIN & EVALUATE
# ============================================================================

def train_and_evaluate(X, y, scaler, feature_names):
    print("\n" + "=" * 60)
    print("  TRAINING LSTM MODEL")
    print("=" * 60)

    # SMOTE for class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  After SMOTE: {X_res.shape[0]} samples (balanced)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Reshape for LSTM: (samples, 1, features)
    X_train_lstm = np.array(X_train, dtype=np.float32).reshape(-1, 1, X_train.shape[1])
    X_test_lstm = np.array(X_test, dtype=np.float32).reshape(-1, 1, X_test.shape[1])
    y_train_np = np.array(y_train, dtype=np.float32)
    y_test_np = np.array(y_test, dtype=np.float32)

    print(f"  Train: {X_train_lstm.shape}  Test: {X_test_lstm.shape}")

    # Build model
    model = build_lstm_model(X_train_lstm.shape[2])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20,
                               restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=7, min_lr=1e-6, verbose=1)

    # Train with more epochs
    history = model.fit(
        X_train_lstm, y_train_np,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_np),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    y_pred_prob = model.predict(X_test_lstm).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred)
    rec = recall_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    roc = roc_auc_score(y_test_np, y_pred_prob)
    cm = confusion_matrix(y_test_np, y_pred)

    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  ROC-AUC:   {roc * 100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    report = classification_report(y_test_np, y_pred, output_dict=True)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test_np, y_pred_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_np, y_pred_prob)

    # Training history
    train_hist = {
        'train_loss': [float(v) for v in history.history['loss']],
        'val_loss': [float(v) for v in history.history['val_loss']],
        'train_accuracy': [float(v) for v in history.history['accuracy']],
        'val_accuracy': [float(v) for v in history.history['val_accuracy']],
    }

    # Metrics dictionary
    metrics = {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc), 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr_curve': {'precision': prec_curve.tolist(), 'recall': rec_curve.tolist()},
        'training_history': train_hist,
    }

    return model, metrics, X_train, X_test, y_test_np, y_pred, y_pred_prob


# ============================================================================
#  5. SHAP EXPLAINABILITY
# ============================================================================

def compute_shap(model, X_train, X_test, feature_names):
    print("\n" + "=" * 60)
    print("  COMPUTING SHAP VALUES (Explainable AI)")
    print("=" * 60)

    import shap

    # Use a subset for background and explanation
    bg_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    explain_sample = X_test.sample(min(50, len(X_test)), random_state=42)

    def model_predict(X_flat):
        X_3d = np.array(X_flat, dtype=np.float32).reshape(-1, 1, X_flat.shape[1])
        return model.predict(X_3d, verbose=0).flatten()

    bg_summary = shap.kmeans(bg_sample, 10)
    explainer = shap.KernelExplainer(model_predict, bg_summary)

    print("  Computing SHAP values for explanation samples...")
    shap_values = explainer.shap_values(explain_sample)

    # Global feature importance (mean absolute SHAP)
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    feature_importance = dict(zip(feature_names, [round(float(v), 6) for v in mean_abs_shap]))

    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(),
                                     key=lambda x: x[1], reverse=True))

    print("\n  Top 10 Feature Importances (SHAP):")
    for i, (feat, imp) in enumerate(list(feature_importance.items())[:10]):
        bar = "#" * int(imp * 200)
        print(f"    {i+1:>2}. {feat:<25} {imp:.6f}  {bar}")

    # Save background data
    bg_sample.to_csv('shap_background.csv', index=False)

    shap_data = {
        'feature_importance': feature_importance,
        'shap_values_sample': shap_vals.tolist(),
        'explain_sample_index': explain_sample.index.tolist(),
    }

    return shap_data


# ============================================================================
#  6. MAIN PIPELINE
# ============================================================================

def main():
    print("\n")
    print("*" * 60)
    print("  CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("  BiLSTM + Attention + Drift Analysis + SHAP")
    print("*" * 60)

    # 1. Load & preprocess
    X, y, scaler, feature_names, raw_df = load_and_preprocess()

    # 2. Behavioral Drift Analysis
    drift_results = compute_drift_analysis(raw_df, feature_names)

    # 3. Train & Evaluate
    model, metrics, X_train, X_test, y_test, y_pred, y_pred_prob = \
        train_and_evaluate(X, y, scaler, feature_names)

    # 4. SHAP Explainability
    shap_data = compute_shap(model, X_train, X_test, feature_names)

    # 5. Save everything
    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)

    model.save('lstm_churn_model.h5')
    print("  [OK] lstm_churn_model.h5")

    joblib.dump(scaler, 'scaler.pkl')
    print("  [OK] scaler.pkl")

    joblib.dump(feature_names, 'feature_names.pkl')
    print("  [OK] feature_names.pkl")

    # Save all metrics, drift, and SHAP data
    all_results = {
        'metrics': metrics,
        'drift_analysis': drift_results,
        'shap_data': {
            'feature_importance': shap_data['feature_importance'],
        },
        'model_info': {
            'architecture': 'Bidirectional LSTM with Self-Attention',
            'layers': ['BiLSTM(128)', 'BatchNorm', 'Dropout(0.3)',
                       'BiLSTM(64)', 'BatchNorm', 'Dropout(0.3)',
                       'Self-Attention', 'Dense(64)', 'Dense(32)', 'Sigmoid'],
            'optimizer': 'Adam',
            'loss': 'binary_crossentropy',
            'n_features': len(feature_names),
            'feature_names': feature_names,
        }
    }

    with open('training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  [OK] training_results.json")

    # Save SHAP values separately (large data)
    joblib.dump(shap_data, 'shap_results.pkl')
    print("  [OK] shap_results.pkl")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC-AUC:   {metrics['roc_auc'] * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
