"""
Extra experiments for the ToN-IoT paper. One run produces:
  1. ablation_dst_port.csv      models with and without the destination port
  2. bootstrap_ci.csv           95% bootstrap confidence intervals on the test set
  3. mcnemar.csv                McNemar tests between pairs of models
  4. per_class_metrics.csv      precision, recall, F1 for each of the ten classes
  5. rare_class_experiment.csv  XGBoost with class weighting, focused on the mitm class
  6. extra_results.csv          main metrics, training and inference time, including a neural network (MLP)

Colab setup (its own cell):   !pip install xgboost
Then (another cell):          %run extra_experiments.py
Keep SAMPLE and SEED the same as in toniot_experiment.py so the test set is identical.
Expected run time on Colab CPU: roughly 15 to 30 minutes. Each CSV is saved as soon as it is ready.
"""
import os
import time
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------- settings ----------------
CSV_PATH = "Train_Test_Network.csv"
OUT_DIR = "."
SEED = 42
SAMPLE = 150000       # must match the main experiment
MAX_CATEGORIES = 30
N_BOOT = 500          # bootstrap resamples
# ------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
ID_COLS = ["ts", "src_ip", "dst_ip", "src_port"]


def save(df, name):
    df.round(4).to_csv(os.path.join(OUT_DIR, name), index=False)
    print(f"  saved {name}")


# ---------- same preprocessing as the main experiment ----------
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.replace("-", np.nan).drop_duplicates()
if SAMPLE and len(df) > SAMPLE:
    df = df.sample(SAMPLE, random_state=SEED)
y_binary = df["label"].astype(int).values
y_type = df["type"].astype(str).values
X = df.drop(columns=["label", "type"] + [c for c in ID_COLS if c in df.columns])


def is_text(col):
    return not pd.api.types.is_numeric_dtype(col)


for c in X.columns:
    if is_text(X[c]):
        conv = pd.to_numeric(X[c], errors="coerce")
        if conv.notna().sum() > 0.9 * X[c].notna().sum() and X[c].notna().sum() > 0:
            X[c] = conv
cat_cols = [c for c in X.columns if is_text(X[c])]
dropped = [c for c in cat_cols if X[c].nunique() > MAX_CATEGORIES]
cat_cols = [c for c in cat_cols if c not in dropped]
X = X.drop(columns=dropped)
for c in cat_cols:
    X[c] = X[c].fillna("none").astype(str)
X = pd.get_dummies(X, columns=cat_cols).fillna(0).astype(float)
X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt_") for c in X.columns]
if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(X.columns, X.columns.duplicated(keep=False)))]
print("Feature matrix:", X.shape)


# ---------- helpers ----------
def make_models(task):
    rf_p = dict(n_estimators=100, max_depth=None if task == "binary" else 20)
    xgb_p = (dict(n_estimators=100, max_depth=6, learning_rate=0.1) if task == "binary"
             else dict(n_estimators=300, max_depth=4, learning_rate=0.2))
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced", **rf_p),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", **xgb_p),
        "MLP (128-64)": Pipeline([("scale", StandardScaler()),
                                  ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                        max_iter=100, random_state=SEED))]),
    }


def score(y, p, task):
    avg = "binary" if task == "binary" else "macro"
    return accuracy_score(y, p), f1_score(y, p, average=avg), matthews_corrcoef(y, p)


F1_NAME = {"binary": "F1", "multiclass": "Macro F1"}
TASKS = {"binary": y_binary, "multiclass": y_type}

main_rows, boot_rows, mc_rows, pc_rows, rare_rows, abl_rows = [], [], [], [], [], []

for task, y_raw in TASKS.items():
    print(f"\n===== {task} =====")
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    classes = list(enc.classes_)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # --- train all models once (needed for items 2, 3, 4, 6) ---
    preds = {}
    for name, model in make_models(task).items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_t = time.time() - t0
        t0 = time.time()
        p = model.predict(X_te)
        infer_ms = (time.time() - t0) / len(X_te) * 1000
        preds[name] = p
        acc, f1, mcc = score(y_te, p, task)
        main_rows.append({"Task": task, "Model": name, "Accuracy": acc, F1_NAME[task]: f1, "MCC": mcc,
                          "Train time (s)": train_t, "Inference (ms/flow)": infer_ms})
        print(f"  {name}: acc {acc:.4f}, {F1_NAME[task]} {f1:.4f}, train {train_t:.1f}s")
    save(pd.DataFrame(main_rows), "extra_results.csv")

    # --- item 2a: bootstrap confidence intervals (same resamples for every model) ---
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_te), size=(N_BOOT, len(y_te)))
    for name, p in preds.items():
        vals = np.array([score(y_te[i], p[i], task) for i in idx])
        for j, metric in enumerate(["Accuracy", F1_NAME[task], "MCC"]):
            lo, hi = np.percentile(vals[:, j], [2.5, 97.5])
            boot_rows.append({"Task": task, "Model": name, "Metric": metric,
                              "Estimate": score(y_te, p, task)[j], "CI low": lo, "CI high": hi})
    save(pd.DataFrame(boot_rows), "bootstrap_ci.csv")

    # --- item 2b: McNemar exact test between every pair of models ---
    for a, b in itertools.combinations(preds, 2):
        ca, cb = preds[a] == y_te, preds[b] == y_te
        only_a = int((ca & ~cb).sum())      # a right, b wrong
        only_b = int((~ca & cb).sum())      # b right, a wrong
        n = only_a + only_b
        pval = binomtest(min(only_a, only_b), n, 0.5).pvalue if n > 0 else 1.0
        mc_rows.append({"Task": task, "Model A": a, "Model B": b, "A right, B wrong": only_a,
                        "B right, A wrong": only_b, "p-value": pval})
    save(pd.DataFrame(mc_rows), "mcnemar.csv")

    # --- item 3: per-class precision, recall, F1 (ten-class task only) ---
    if task == "multiclass":
        for name, p in preds.items():
            pr, rc, f1, sup = precision_recall_fscore_support(y_te, p, labels=range(len(classes)), zero_division=0)
            for k, cls in enumerate(classes):
                pc_rows.append({"Model": name, "Class": cls, "Precision": pr[k], "Recall": rc[k],
                                "F1": f1[k], "Support": int(sup[k])})
        save(pd.DataFrame(pc_rows), "per_class_metrics.csv")

    # --- item 1: ablation, remove the destination port ---
    if "dst_port" in X.columns:
        Xtr_d, Xte_d = X_tr.drop(columns=["dst_port"]), X_te.drop(columns=["dst_port"])
        for name in ["Random Forest", "XGBoost"]:
            m = make_models(task)[name]
            p_kept = preds[name]
            p_drop = m.fit(Xtr_d, y_tr).predict(Xte_d)
            for label, p in [("kept", p_kept), ("removed", p_drop)]:
                acc, f1, mcc = score(y_te, p, task)
                abl_rows.append({"Task": task, "Model": name, "dst_port": label,
                                 "Accuracy": acc, F1_NAME[task]: f1, "MCC": mcc})
            print(f"  ablation {name}: acc kept {accuracy_score(y_te, p_kept):.4f}, removed {accuracy_score(y_te, p_drop):.4f}")
        save(pd.DataFrame(abl_rows), "ablation_dst_port.csv")

    # --- item 4: help the rare class with class weighting (ten-class task only) ---
    if task == "multiclass" and "mitm" in classes:
        mi = classes.index("mitm")
        balanced = compute_sample_weight("balanced", y_tr)
        variants = {"XGBoost, no weights": None,
                    "XGBoost, sqrt-balanced weights": np.sqrt(balanced),
                    "XGBoost, balanced weights": balanced}
        for label, w in variants.items():
            m = make_models(task)["XGBoost"]
            p = preds["XGBoost"] if w is None else m.fit(X_tr, y_tr, sample_weight=w).predict(X_te)
            acc, f1, mcc = score(y_te, p, task)
            pr, rc, f1c, _ = precision_recall_fscore_support(y_te, p, labels=[mi], zero_division=0)
            rare_rows.append({"Variant": label, "Accuracy": acc, "Macro F1": f1, "MCC": mcc,
                              "mitm precision": pr[0], "mitm recall": rc[0], "mitm F1": f1c[0]})
            print(f"  {label}: macro F1 {f1:.4f}, mitm recall {rc[0]:.3f}, mitm precision {pr[0]:.3f}")
        save(pd.DataFrame(rare_rows), "rare_class_experiment.csv")

print("\nDone. Download these files from the Files panel:")
print("  ablation_dst_port.csv, bootstrap_ci.csv, mcnemar.csv, per_class_metrics.csv,")
print("  rare_class_experiment.csv, extra_results.csv")
