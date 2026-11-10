"""Fit a depth-3 no-show tree and score it against a show-everyone baseline."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv("clinic_appointments.csv")
n_noshow = int((df["no_show"] == "no-show").sum())
print("shape:", df.shape)
print(f"no-show rows: {n_noshow}")
print(f"no-show share: {n_noshow / len(df):.4f}")

y = (df["no_show"] == "no-show").astype(int)
X = df.drop(columns=["no_show"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

v1 = DecisionTreeClassifier(max_depth=3, random_state=42)
v1.fit(X_train, y_train)

print(export_text(v1, feature_names=list(X_train.columns), show_weights=True))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    v1,
    filled=True,
    feature_names=list(X_train.columns),
    class_names=["show", "no-show"],
    fontsize=9,
    ax=ax,
)
fig.tight_layout()
fig.savefig("tree_v1.png")
plt.close(fig)


def report(name, model_or_preds, X, y):
    if hasattr(model_or_preds, "predict"):
        preds = model_or_preds.predict(X)
    else:
        preds = model_or_preds
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, pos_label=1, zero_division=0)
    recall = recall_score(y, preds, pos_label=1, zero_division=0)
    print(
        f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} "
        f"accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f}"
    )
    return tn, fp, fn, tp


def value(tn, fp, fn, tp):
    return -60 * fp - 120 * fn


baseline = report("baseline test", [0] * len(y_test), X_test, y_test)
report("v1 train", v1, X_train, y_train)
v1_test = report("v1 test", v1, X_test, y_test)

n_test = len(y_test)
for label, counts in [("baseline test", baseline), ("v1 test", v1_test)]:
    total = value(*counts)
    print(f"{label} value: total={total} per appointment={total / n_test:.3f}")


def decide(appointment):
    row = pd.DataFrame([appointment], columns=X_train.columns)
    pred = int(v1.predict(row)[0])
    leaf = int(v1.apply(row)[0])
    p_noshow = float(v1.predict_proba(row)[0, list(v1.classes_).index(1)])
    action = "overbook" if pred == 1 else "hold"
    return action, leaf, p_noshow


D = {
    "lead_time_days": 21,
    "prior_no_shows": 1,
    "age": 27,
    "distance_miles": 9.5,
    "sms_reminder": 0,
    "new_patient": 1,
    "appointment_hour": 15,
}
E = {
    "lead_time_days": 3,
    "prior_no_shows": 0,
    "age": 61,
    "distance_miles": 2.0,
    "sms_reminder": 1,
    "new_patient": 0,
    "appointment_hour": 9,
}

for label, appointment in [("D", D), ("E", E)]:
    action, leaf, p_noshow = decide(appointment)
    print(f"{label}: {action} leaf={leaf} no-show probability={p_noshow:.4f}")

HOURS_PER_DAY = 9
APPOINTMENTS_PER_HOUR = 3
DOCTORS = 3
DAYS_PER_YEAR = 250
APPOINTMENTS_PER_YEAR = HOURS_PER_DAY * APPOINTMENTS_PER_HOUR * DOCTORS * DAYS_PER_YEAR
print("APPOINTMENTS_PER_YEAR:", APPOINTMENTS_PER_YEAR)

baseline_per = value(*baseline) / n_test
v1_per = value(*v1_test) / n_test
baseline_yearly = round(baseline_per * APPOINTMENTS_PER_YEAR)
v1_yearly = round(v1_per * APPOINTMENTS_PER_YEAR)
yearly_savings = v1_yearly - baseline_yearly
print(f"baseline yearly value: {baseline_yearly}")
print(f"v1 yearly value: {v1_yearly}")
print(f"yearly savings: {yearly_savings}")
