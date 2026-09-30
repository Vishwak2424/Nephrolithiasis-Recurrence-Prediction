
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

DATA = Path("data/nephrolithiasis.csv")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

FEATURES = [
    "age","gender","bmi","hypertension","diabetes","previous_stones","family_history",
    "alcohol_consumption","current_medications","daily_water_intake","daily_citrus_intake",
    "daily_salt_intake","normal_calcium_diet","high_potassium_diet","serum_calcium","urine_ph",
    "serum_uric_acid","serum_ipth","serum_creatinine","serum_vitamin_d","urine_24h_calcium",
    "urine_24h_phosphorous","urine_24h_oxalate","urine_24h_uric_acid","urine_24h_citrate",
    "renal_calculi_treatment","surgery_type","stone_burden"
]
CAT = ["gender","alcohol_consumption","renal_calculi_treatment","surgery_type"]
NUM = [c for c in FEATURES if c not in CAT]

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUM),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), CAT)
    ])

def make_pipeline(est):
    return Pipeline([("pre", make_preprocessor()), ("model", est)])

def specificity(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return tn/(tn+fp) if tn+fp else 0.0

def evaluate(name, model, X, y):
    p = model.predict(X)
    s = model.predict_proba(X)[:,1]
    return {
        "model": name,
        "accuracy": accuracy_score(y,p),
        "precision": precision_score(y,p,zero_division=0),
        "recall": recall_score(y,p,zero_division=0),
        "specificity": specificity(y,p),
        "f1": f1_score(y,p,zero_division=0),
        "roc_auc": roc_auc_score(y,s),
        "pr_auc": average_precision_score(y,s)
    }, p, s

def make_eda(df):
    # Missingness
    miss = df[FEATURES + ["recurrence"]].isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    pd.DataFrame({"feature": miss.index, "missing_count": miss.values,
                  "missing_percent": (miss.values/len(df))*100}).to_csv(
        ART/"data_quality.csv", index=False
    )

    # Class balance
    counts = df["recurrence"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["No recurrence","Recurrence"], [counts.get(0,0),counts.get(1,0)])
    ax.set_ylabel("Number of records"); ax.set_title("Target Class Distribution")
    for i,v in enumerate([counts.get(0,0),counts.get(1,0)]): ax.text(i,v+10,str(v),ha="center")
    fig.tight_layout(); fig.savefig(ART/"eda_class_balance.png", dpi=180); plt.close(fig)

    # Missingness
    fig, ax = plt.subplots(figsize=(8,5))
    if len(miss):
        top = miss.head(15).sort_values()
        ax.barh(top.index, top.values)
        ax.set_xlabel("Missing values"); ax.set_title("Top Features by Missing Values")
    else:
        ax.text(.5,.5,"No missing values",ha="center",va="center")
    fig.tight_layout(); fig.savefig(ART/"eda_missingness.png", dpi=180); plt.close(fig)

    # Numeric distributions
    selected = ["age","bmi","daily_water_intake","urine_ph","serum_calcium",
                "urine_24h_oxalate","urine_24h_citrate","stone_burden"]
    fig, axes = plt.subplots(2,4,figsize=(13,7))
    for ax,col in zip(axes.ravel(),selected):
        ax.hist(df[col].dropna(),bins=25)
        ax.set_title(col.replace("_"," ").title())
    fig.suptitle("Clinical Feature Distributions", y=1.02)
    fig.tight_layout(); fig.savefig(ART/"eda_distributions.png",dpi=180); plt.close(fig)

    # Correlation heatmap for numeric variables
    corr_cols = [c for c in NUM if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[corr_cols + ["recurrence"]].corr()
    fig, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout(); fig.savefig(ART/"eda_correlation.png",dpi=180); plt.close(fig)

def main():
    if not DATA.exists():
        raise FileNotFoundError("data/nephrolithiasis.csv is missing.")
    df = pd.read_csv(DATA)
    required = FEATURES + ["recurrence"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing columns: " + str(missing_cols))

    df["recurrence"] = df["recurrence"].astype(int)
    X, y = df[FEATURES], df["recurrence"]
    make_eda(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3,
                                                class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                                min_samples_leaf=2, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                            min_samples_leaf=2, class_weight="balanced",
                                            n_jobs=-1, random_state=42),
        "SVM": SVC(C=2, probability=True, class_weight="balanced", random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.05, max_leaf_nodes=20, random_state=42)
    }

    # Hyperparameter tuning for the main linear model.
    base_lr = make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"))
    grid = GridSearchCV(
        base_lr,
        {"model__C":[0.01,0.05,0.1,0.25,0.5,1,2,5,10]},
        cv=StratifiedKFold(5,shuffle=True,random_state=42),
        scoring="roc_auc", n_jobs=-1, return_train_score=True
    )
    grid.fit(Xtr,ytr)
    best_lr = grid.best_estimator_
    pd.DataFrame(grid.cv_results_)[
        ["param_model__C","mean_test_score","std_test_score","rank_test_score"]
    ].sort_values("rank_test_score").to_csv(ART/"hyperparameter_tuning.csv",index=False)
    models["Tuned Logistic Regression"] = best_lr

    results=[]; fitted={}
    for name, est in models.items():
        model = est if name == "Tuned Logistic Regression" else make_pipeline(est)
        model.fit(Xtr,ytr)
        r, p, s = evaluate(name,model,Xte,yte)
        results.append(r); fitted[name]=model

    results_df = pd.DataFrame(results).sort_values("accuracy",ascending=False)
    results_df.to_csv(ART/"metrics.csv",index=False)

    # Cross-validation for every model, not just the winner.
    cv = StratifiedKFold(5,shuffle=True,random_state=42)
    cv_rows=[]
    for name,model in fitted.items():
        scores = cross_validate(model, X, y, cv=cv,
                                scoring=["accuracy","precision","recall","f1","roc_auc"],
                                n_jobs=-1)
        cv_rows.append({
            "model":name,
            "accuracy_mean":scores["test_accuracy"].mean(),
            "accuracy_std":scores["test_accuracy"].std(),
            "precision_mean":scores["test_precision"].mean(),
            "recall_mean":scores["test_recall"].mean(),
            "f1_mean":scores["test_f1"].mean(),
            "roc_auc_mean":scores["test_roc_auc"].mean()
        })
    cv_df=pd.DataFrame(cv_rows).sort_values("accuracy_mean",ascending=False)
    cv_df.to_csv(ART/"cross_validation_models.csv",index=False)

    best_name = results_df.iloc[0]["model"]
    best = fitted[best_name]
    joblib.dump(best,ART/"best_model.joblib")

    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "categorical_features": CAT,
        "numeric_features": NUM,
        "synthetic_demo": True,
        "train_test_split": "80/20 stratified, random_state=42",
        "cv": "5-fold stratified",
        "tuned_model": "Logistic Regression C grid searched by ROC-AUC"
    }
    (ART/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

    # Best-model holdout curves and confusion matrix.
    r,p,s = evaluate(best_name,best,Xte,yte)
    cm=confusion_matrix(yte,p)
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["No Recurrence","Recurrence"],
                yticklabels=["No Recurrence","Recurrence"],ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix - "+best_name)
    fig.tight_layout(); fig.savefig(ART/"confusion_matrix.png",dpi=180); plt.close(fig)

    fpr,tpr,_=roc_curve(yte,s)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,label=f"AUC = {r['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],"--"); ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve - "+best_name); ax.legend()
    fig.tight_layout(); fig.savefig(ART/"roc_curve.png",dpi=180); plt.close(fig)

    precision,recall,_=precision_recall_curve(yte,s)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(recall,precision,label=f"AP = {r['pr_auc']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve - "+best_name); ax.legend()
    fig.tight_layout(); fig.savefig(ART/"pr_curve.png",dpi=180); plt.close(fig)

    # Global permutation importance.
    pi=permutation_importance(best,Xte,yte,n_repeats=10,random_state=42,scoring="accuracy")
    pd.DataFrame({"feature":FEATURES,"importance_mean":pi.importances_mean,
                  "importance_std":pi.importances_std}).sort_values(
        "importance_mean",ascending=False).to_csv(ART/"feature_importance.csv",index=False)

    print("\nMODEL RESULTS")
    print(results_df.to_string(index=False))
    print("\nBEST:",best_name)
    print("Tuned LR best C:",grid.best_params_["model__C"])
    print("\n5-FOLD CROSS-VALIDATION")
    print(cv_df.to_string(index=False))
    print("\nNOTE: Current dataset is synthetic demonstration data.")
    print("Do not present these metrics as clinical evidence.")

if __name__=="__main__":
    main()
