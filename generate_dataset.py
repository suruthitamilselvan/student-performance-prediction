import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# ── Features ────────────────────────────────────────────────
study_hours       = np.random.uniform(1, 10, n)
attendance        = np.random.uniform(50, 100, n)
previous_marks    = np.random.uniform(30, 100, n)
assignments_done  = np.random.randint(0, 11, n)          # out of 10
sleep_hours       = np.random.uniform(4, 10, n)
internet_hours    = np.random.uniform(0, 8, n)

# ── Target: final_score (realistic weighted formula + noise) ─
final_score = (
    study_hours      * 3.5 +
    attendance       * 0.3 +
    previous_marks   * 0.4 +
    assignments_done * 1.5 +
    sleep_hours      * 1.2 -
    internet_hours   * 1.0 +
    np.random.normal(0, 5, n)          # noise
)

# Clip to 0–100 range
final_score = np.clip(final_score, 0, 100).round(2)

# ── Target: pass_fail (pass if score >= 40) ──────────────────
pass_fail = (final_score >= 40).astype(int)   # 1 = Pass, 0 = Fail

# ── Build DataFrame ──────────────────────────────────────────
df = pd.DataFrame({
    'study_hours'      : study_hours.round(2),
    'attendance_pct'   : attendance.round(2),
    'previous_marks'   : previous_marks.round(2),
    'assignments_done' : assignments_done,
    'sleep_hours'      : sleep_hours.round(2),
    'internet_hours'   : internet_hours.round(2),
    'final_score'      : final_score,
    'pass_fail'        : pass_fail
})

# ── Save ─────────────────────────────────────────────────────
df.to_csv('student_data.csv', index=False)

print("Dataset created! Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic stats:")
print(df.describe().round(2))
print("\nPass/Fail distribution:")
print(df['pass_fail'].value_counts().rename({1:'Pass', 0:'Fail'}))