"""
IoT intrusion detection on the ToN-IoT network dataset (Train_Test_Network.csv).
Compares Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting and XGBoost,
for both binary (normal vs attack) and multi-class (attack type) detection.

Colab setup (run in a cell first):
    !pip install xgboost

Then upload Train_Test_Network.csv to Colab and run this file:
    %run toniot_experiment.py

Outputs (in the same folder):
    results_binary.csv, results_multiclass.csv
    cm_binary.png, cm_multiclass.png
    feature_importance.png
"""
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------- settings ----------------
CSV_PATH = "Train_Test_Network.csv"
SEED = 42
SAMPLE = 150000       # rows to use while testing. Set to None for the final full run.
TUNE_ROWS = 40000     # training rows used for hyperparameter search and cross-validation
MAX_CATEGORIES = 30   # categorical columns with more distinct values than this get dropped
# ------------------------------------------

# Columns that identify who sent traffic rather than how it behaves. The attacker machines
# have fixed IPs in this dataset, so keeping them lets a model "cheat". Drop them.
ID_COLS = ["ts", "src_ip", "dst_ip", "src_port"]

# ---------- 1. load and clean ----------
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Raw shape:", df.shape)
df = df.replace("-", np.nan)
df = df.drop_duplicates()
if SAMPLE and len(df) > SAMPLE:
    df = df.sample(SAMPLE, random_state=SEED)
print("After dedupe/sample:", df.shape)
print("\nBinary label counts:\n", df["label"].value_counts())
print("\nAttack type counts:\n", df["type"].value_counts())

y_binary = df["label"].astype(int).values
y_type = df["type"].astype(str).values
X = df.drop(columns=["label", "type"] + [c for c in ID_COLS if c in df.columns])

# numeric-looking text columns become numbers, the rest stay categorical
def is_text(col):
    return not pd.api.types.is_numeric_dtype(col)

for c in X.columns:
    if is_text(X[c]):
        converted = pd.to_numeric(X[c], errors="coerce")
        if converted.notna().sum() > 0.9 * X[c].notna().sum() and X[c].notna().sum() > 0:
            X[c] = converted

cat_cols = [c for c in X.columns if is_text(X[c])]
dropped = [c for c in cat_cols if X[c].nunique() > MAX_CATEGORIES]
cat_cols = [c for c in cat_cols if c not in dropped]
X = X.drop(columns=dropped)
print("\nDropped high-cardinality text columns:", dropped)

for c in cat_cols:
    X[c] = X[c].fillna("none").astype(str)
X = pd.get_dummies(X, columns=cat_cols)
X = X.fillna(0).astype(float)
print("Final feature matrix:", X.shape)

# ---------- 2. models and search spaces ----------
def make_models():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1),
                {"clf__n_neighbors": [3, 5, 7, 11]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
                          {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 5, 10]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced"),
                          {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 20, None]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1, 0.2],
                               "clf__max_depth": [3, 5]}),
        "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist"),
                    {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.05, 0.1, 0.2],
                     "clf__max_depth": [4, 6, 8]}),
    }


def run_task(task, y_raw):
    """Run all six models on one labelling (binary or multi-class)."""
    multiclass = task == "multiclass"
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    class_names = list(enc.classes_)
    avg = "weighted" if multiclass else "binary"

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    # smaller stratified slice of the training set for tuning and CV (keeps runtime sane)
    if len(X_tr) > TUNE_ROWS:
        X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=TUNE_ROWS, stratify=y_tr, random_state=SEED)
    else:
        X_sub, y_sub = X_tr, y_tr

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, matrices, fitted = [], {}, {}

    for name, (clf, grid) in make_models().items():
        print(f"\n[{task}] {name}")
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(pipe, grid, n_iter=min(6, n_combos), cv=3,
                                    scoring="f1_weighted", random_state=SEED, n_jobs=-1)
        search.fit(X_sub, y_sub)

        cv_scores = cross_val_score(clone(pipe).set_params(**search.best_params_),
                                    X_sub, y_sub, cv=cv, scoring="f1_weighted", n_jobs=-1)

        best = clone(pipe).set_params(**search.best_params_)
        t0 = time.time()
        best.fit(X_tr, y_tr)
        train_time = time.time() - t0

        t0 = time.time()
        pred = best.predict(X_te)
        infer_ms = (time.time() - t0) / len(X_te) * 1000

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
            "F1": f1_score(y_te, pred, average=avg, zero_division=0),
            "MCC": matthews_corrcoef(y_te, pred),
            "CV F1 mean": cv_scores.mean(),
            "CV F1 std": cv_scores.std(),
            "Train time (s)": train_time,
            "Inference (ms/sample)": infer_ms,
            "Best params": str(search.best_params_),
        }
        if not multiclass:
            row["ROC-AUC"] = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
        else:
            row["Macro F1"] = f1_score(y_te, pred, average="macro", zero_division=0)
        rows.append(row)
        matrices[name] = confusion_matrix(y_te, pred)
        fitted[name] = best
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "Best params"})

    results = pd.DataFrame(rows)
    num_cols = results.select_dtypes("number").columns
    results[num_cols] = results[num_cols].round(4)
    results.to_csv(f"results_{task}.csv", index=False)
    print(f"\n=== {task} results ===")
    print(results.drop(columns=["Best params"]).to_string(index=False))

    # confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18 if multiclass else 15, 11 if multiclass else 9))
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7 if multiclass else 10})
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if multiclass:
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(f"cm_{task}.png", dpi=200)
    plt.close()
    return results, fitted


# ---------- 3. run both tasks ----------
res_bin, fitted_bin = run_task("binary", y_binary)
res_multi, _ = run_task("multiclass", y_type)

# ---------- 4. feature importance from Random Forest (binary task) ----------
rf = fitted_bin["Random Forest"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
imp[::-1].plot(kind="barh", color="seagreen")
plt.xlabel("Importance")
plt.title("Top 15 features, Random Forest (binary)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
print("\nTop features:\n", imp.round(4))
print("\nDone. Files written: results_binary.csv, results_multiclass.csv, cm_binary.png, cm_multiclass.png, feature_importance.png")
