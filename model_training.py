import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load processed data ───────────────────────────────────────
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_r = data['X_train_r']
X_test_r  = data['X_test_r']
y_train_r = data['y_train_r']
y_test_r  = data['y_test_r']

X_train_c = data['X_train_c']
X_test_c  = data['X_test_c']
y_train_c = data['y_train_c']
y_test_c  = data['y_test_c']

features  = data['features']

# ════════════════════════════════════════════════════════════
# MODEL 1 — Linear Regression (predict final score)
# ════════════════════════════════════════════════════════════
print("=" * 50)
print("MODEL 1 - LINEAR REGRESSION")
print("=" * 50)

lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
y_pred_lr = lr.predict(X_test_r)

mae_lr  = mean_absolute_error(y_test_r, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_r, y_pred_lr))
r2_lr   = r2_score(y_test_r, y_pred_lr)

print("MAE  :", round(mae_lr, 2))
print("RMSE :", round(rmse_lr, 2))
print("R2   :", round(r2_lr, 2))

# Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test_r, y_pred_lr, alpha=0.5, color='steelblue')
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--')
plt.title('Linear Regression - Actual vs Predicted')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.tight_layout()
plt.savefig('model1_actual_vs_predicted.png')
plt.show()
print("Plot saved!")

# ════════════════════════════════════════════════════════════
# MODEL 2 — Decision Tree (predict pass/fail)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("MODEL 2 - DECISION TREE")
print("=" * 50)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_c, y_train_c)
y_pred_dt = dt.predict(X_test_c)

acc_dt = accuracy_score(y_test_c, y_pred_dt)
print("Accuracy:", round(acc_dt * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_dt,
      target_names=['Fail', 'Pass']))

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_c, y_pred_dt),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail','Pass'],
            yticklabels=['Fail','Pass'])
plt.title('Decision Tree - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model2_confusion_matrix.png')
plt.show()
print("Plot saved!")

# ════════════════════════════════════════════════════════════
# MODEL 3 — Random Forest (predict pass/fail)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("MODEL 3 - RANDOM FOREST")
print("=" * 50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_c, y_train_c)
y_pred_rf = rf.predict(X_test_c)

acc_rf = accuracy_score(y_test_c, y_pred_rf)
print("Accuracy:", round(acc_rf * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_rf,
      target_names=['Fail', 'Pass']))

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_c, y_pred_rf),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=['Fail','Pass'],
            yticklabels=['Fail','Pass'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model3_confusion_matrix.png')
plt.show()
print("Plot saved!")

# ════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — Random Forest
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE - RANDOM FOREST")
print("=" * 50)

importances = rf.feature_importances_
indices     = np.argsort(importances)[::-1]

print("Feature ranking:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {features[idx]} : {round(importances[idx], 4)}")

plt.figure(figsize=(8, 5))
plt.bar(range(len(features)),
        importances[indices], color='steelblue')
plt.xticks(range(len(features)),
           [features[i] for i in indices], rotation=15)
plt.title('Feature Importance - Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('model4_feature_importance.png')
plt.show()
print("Plot saved!")

# ════════════════════════════════════════════════════════════
# SAVE ALL MODELS
# ════════════════════════════════════════════════════════════
with open('model_lr.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('model_dt.pkl', 'wb') as f:
    pickle.dump(dt, f)

with open('model_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("\n" + "=" * 50)
print("ALL 3 MODELS TRAINED AND SAVED!")
print("  model_lr.pkl  - Linear Regression")
print("  model_dt.pkl  - Decision Tree")
print("  model_rf.pkl  - Random Forest")
print("=" * 50)