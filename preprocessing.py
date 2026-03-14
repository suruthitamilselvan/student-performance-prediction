import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('student_data.csv')

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# ── Features and Targets ─────────────────────────────────────
features = ['study_hours', 'attendance_pct', 'previous_marks',
            'assignments_done', 'sleep_hours', 'internet_hours']

X = df[features]
y_regression     = df['final_score']   # for Linear Regression
y_classification = df['pass_fail']     # for Decision Tree / Random Forest

# ── Train Test Split ─────────────────────────────────────────
# Regression split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_regression, test_size=0.2, random_state=42)

# Classification split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_classification, test_size=0.2, random_state=42)

print("\nRegression split:")
print("  Train size:", X_train_r.shape)
print("  Test size :", X_test_r.shape)

print("\nClassification split:")
print("  Train size:", X_train_c.shape)
print("  Test size :", X_test_c.shape)

# ── Feature Scaling ───────────────────────────────────────────
scaler = StandardScaler()

X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled  = scaler.transform(X_test_r)

X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled  = scaler.transform(X_test_c)

print("\nScaling done!")
print("Sample scaled values (first row):")
print(X_train_r_scaled[0])

# ── Save processed data ───────────────────────────────────────
processed = {
    'X_train_r': X_train_r_scaled,
    'X_test_r' : X_test_r_scaled,
    'y_train_r': y_train_r.values,
    'y_test_r' : y_test_r.values,

    'X_train_c': X_train_c_scaled,
    'X_test_c' : X_test_c_scaled,
    'y_train_c': y_train_c.values,
    'y_test_c' : y_test_c.values,

    'scaler'   : scaler,
    'features' : features
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed, f)

print("\nPreprocessing complete!")
print("processed_data.pkl saved!")