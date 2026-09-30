
from pathlib import Path
import sqlite3
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from rag import LocalRAG, ollama

ART=Path("artifacts")
MODEL=ART/"best_model.joblib"
MET=ART/"metrics.csv"
CV=ART/"cross_validation.csv"
CVALL=ART/"cross_validation_models.csv"
FI=ART/"feature_importance.csv"
DATA=Path("data/nephrolithiasis.csv")

FEATURES=[
"age","gender","bmi","hypertension","diabetes","previous_stones","family_history",
"alcohol_consumption","current_medications","daily_water_intake","daily_citrus_intake",
"daily_salt_intake","normal_calcium_diet","high_potassium_diet","serum_calcium","urine_ph",
"serum_uric_acid","serum_ipth","serum_creatinine","serum_vitamin_d","urine_24h_calcium",
"urine_24h_phosphorous","urine_24h_oxalate","urine_24h_uric_acid","urine_24h_citrate",
"renal_calculi_treatment","surgery_type","stone_burden"]

rag=LocalRAG()

def db():
    c=sqlite3.connect("predictions.db")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY, patient TEXT, prediction INTEGER,
        probability REAL, time TEXT)""")
    c.commit()
    return c
db().close()

def metrics():
    return pd.read_csv(MET) if MET.exists() else pd.DataFrame()

def cards():
    d=metrics()
    if d.empty: return "No metrics available. Run training first."
    b=d.sort_values("accuracy",ascending=False).iloc[0]
    return f"""### 🏆 Best Model: **{b['model']}**

| Metric | Score |
|---|---:|
| Accuracy | **{b.accuracy:.2%}** |
| Precision | {b.precision:.2%} |
| Recall | {b.recall:.2%} |
| Specificity | {b.specificity:.2%} |
| F1 Score | {b.f1:.2%} |
| ROC-AUC | {b.roc_auc:.2%} |
| PR-AUC | {b.pr_auc:.2%} |

> 
"""

def accplot():
    d=metrics(); fig,ax=plt.subplots(figsize=(8,5))
    if d.empty: ax.text(.5,.5,"No metrics",ha="center")
    else:
        d=d.sort_values("accuracy")
        ax.barh(d.model,d.accuracy*100)
        ax.set(xlabel="Accuracy (%)",title="Model Accuracy Comparison",xlim=(0,100))
        for i,v in enumerate(d.accuracy*100): ax.text(v+.3,i,f"{v:.2f}%")
    fig.tight_layout(); return fig

def multplot():
    d=metrics(); fig,ax=plt.subplots(figsize=(9,5))
    if not d.empty:
        x=np.arange(len(d)); w=.18
        for j,c in enumerate(["accuracy","precision","recall","f1"]):
            ax.bar(x+(j-1.5)*w,d[c]*100,w,label=c.replace("_"," ").title())
        ax.set_xticks(x); ax.set_xticklabels(d.model,rotation=25,ha="right")
        ax.set_ylim(0,100); ax.set_ylabel("Score (%)")
        ax.set_title("Accuracy vs Precision vs Recall vs F1"); ax.legend()
    fig.tight_layout(); return fig

def cvplot():
    d=pd.read_csv(CVALL) if CVALL.exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(9,5))
    if not d.empty and {"model","accuracy_mean"}.issubset(d.columns):
        d=d.sort_values("accuracy_mean")
        err=d["accuracy_std"]*100 if "accuracy_std" in d.columns else None
        ax.barh(d.model,d.accuracy_mean*100,xerr=err)
        ax.set(xlabel="5-Fold CV Accuracy (%)",title="Cross-Validation Model Stability",xlim=(0,100))
        for i,v in enumerate(d.accuracy_mean*100): ax.text(v+.3,i,f"{v:.2f}%")
    else: ax.text(.5,.5,"No model-level CV results. Run train.py.",ha="center",va="center")
    fig.tight_layout(); return fig

def fiplot():
    d=pd.read_csv(FI) if FI.exists() else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(8,6))
    if not d.empty and "feature" in d.columns:
        # Support both current artifacts (importance_mean) and older packaged artifacts (importance).
        col="importance_mean" if "importance_mean" in d.columns else ("importance" if "importance" in d.columns else None)
        if col is not None:
            d=d.sort_values(col).tail(15)
            ax.barh(d.feature,d[col])
            ax.set(xlabel="Permutation Importance",title="Top 15 Global Feature Importance")
        else:
            ax.text(.5,.5,"Feature importance column missing",ha="center",va="center")
    else: ax.text(.5,.5,"No feature importance",ha="center",va="center")
    fig.tight_layout(); return fig

def cmplot():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        d=pd.read_csv(DATA); X=d[FEATURES]; y=d.recurrence.astype(int)
        _,xt,_,yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        p=joblib.load(MODEL).predict(xt); cm=confusion_matrix(yt,p,labels=[0,1])
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm)
        ax.set_xticks([0,1]);ax.set_yticks([0,1])
        ax.set_xticklabels(["No Recurrence","Recurrence"]);ax.set_yticklabels(["No Recurrence","Recurrence"])
        ax.set(xlabel="Predicted",ylabel="Actual",title="Confusion Matrix - Best Model")
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        fig.colorbar(im,ax=ax);fig.tight_layout();return fig
    except Exception as e:
        fig,ax=plt.subplots();ax.text(.5,.5,str(e),ha="center",wrap=True);ax.axis("off");return fig

def curveplot(kind="roc"):
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
        d=pd.read_csv(DATA);X=d[FEATURES];y=d.recurrence.astype(int)
        _,xt,_,yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        m=joblib.load(MODEL);s=m.predict_proba(xt)[:,1]
        fig,ax=plt.subplots(figsize=(7,5))
        if kind=="roc":
            x,z,_=roc_curve(yt,s); score=roc_auc_score(yt,s)
            ax.plot(x,z,label=f"AUC={score:.3f}");ax.plot([0,1],[0,1],"--")
            ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",title="ROC Curve");ax.legend()
        else:
            z,x,_=precision_recall_curve(yt,s);score=average_precision_score(yt,s)
            ax.plot(x,z,label=f"AP={score:.3f}")
            ax.set(xlabel="Recall",ylabel="Precision",title="Precision-Recall Curve");ax.legend()
        fig.tight_layout();return fig
    except Exception as e:
        fig,ax=plt.subplots();ax.text(.5,.5,str(e),ha="center",wrap=True);ax.axis("off");return fig

def eda_class():
    d=pd.read_csv(DATA); c=d.recurrence.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(6,4))
    vals=[c.get(0,0),c.get(1,0)]
    ax.bar(["No Recurrence","Recurrence"],vals)
    ax.set_ylabel("Records");ax.set_title("Target Class Distribution")
    for i,v in enumerate(vals): ax.text(i,v+10,str(v),ha="center")
    fig.tight_layout();return fig

def eda_missing():
    d=pd.read_csv(DATA); m=d[FEATURES].isna().sum().sort_values(ascending=False).head(15)
    fig,ax=plt.subplots(figsize=(8,5))
    m=m[m>0].sort_values()
    if len(m): ax.barh(m.index,m.values);ax.set_xlabel("Missing values")
    else: ax.text(.5,.5,"No missing values",ha="center")
    ax.set_title("Missing Data Overview");fig.tight_layout();return fig

def eda_dist():
    d=pd.read_csv(DATA)
    cols=["age","bmi","daily_water_intake","urine_ph","serum_calcium",
          "urine_24h_oxalate","urine_24h_citrate","stone_burden"]
    fig,axes=plt.subplots(2,4,figsize=(13,7))
    for ax,col in zip(axes.ravel(),cols):
        ax.hist(d[col].dropna(),bins=25);ax.set_title(col.replace("_"," ").title())
    fig.suptitle("Clinical Feature Distributions",y=1.02);fig.tight_layout();return fig

def history():
    c=db()
    d=pd.read_sql_query("select id,patient,prediction,probability,time from predictions order by id desc limit 20",c)
    c.close()
    if not d.empty:
        d["prediction"]=d.prediction.map({0:"Lower Risk",1:"Higher Risk"})
        d["probability"]=(d.probability*100).round(2).astype(str)+"%"
    return d

def local_explanation(row, model, probability):
    """Return a local, model-specific explanation and the strongest feature names.

    For logistic regression, contribution = transformed feature value * coefficient.
    For other models, global permutation importance is used as a transparent fallback.
    These are model explanations, not medical causation claims.
    """
    try:
        if "Logistic Regression" not in str(model):
            raise ValueError("Non-linear best model")
        pre=model.named_steps["pre"]; clf=model.named_steps["model"]
        z=pre.transform(row)
        names=pre.get_feature_names_out()
        coefs=clf.coef_[0]
        vals=np.asarray(z)[0]
        contrib=vals*coefs
        grouped={}
        for n,v in zip(names,contrib):
            key=n.split("__",1)[-1]
            for f in FEATURES:
                if key==f or key.startswith(f+"_"):
                    grouped[f]=grouped.get(f,0)+float(v); break
        top=sorted(grouped.items(),key=lambda x:abs(x[1]),reverse=True)[:6]
        lines=["### 🧠 Why did the model make this prediction?"]
        if not top:
            lines.append("No local feature contributions were available.")
        else:
            for f,v in top:
                direction="increased" if v>0 else "reduced"
                lines.append(f"- **{f.replace('_',' ').title()}** {direction} the model score ({v:+.3f}).")
        lines.append(f"\nModel probability: **{probability:.1%}**. These are model contributions, not medical causation.")
        return "\n".join(lines), [f for f,_ in top]
    except Exception:
        fi=pd.read_csv(FI).sort_values("importance_mean",ascending=False).head(6)
        features=fi.feature.tolist()
        text="### 🧠 Global model drivers\n" + "\n".join(
            f"- **{r.feature.replace('_',' ').title()}** — importance {r.importance_mean:.4f}"
            for _,r in fi.iterrows()
        ) + "\n\nThese show model importance, not medical causation."
        return text, features

def format_evidence(results):
    if not results:
        return "### 🔎 Retrieved Evidence\nNo query was supplied."
    lines=["### 🔎 Retrieved Evidence", "The results below are ranked by TF-IDF similarity to your query."]
    for r in results:
        lines += [
            f"#### {r['rank']}. {r['title']}",
            f"**Retrieval score:** `{r['score']:.3f}`",
            r["text"],
            ""
        ]
    return "\n".join(lines)

def build_rag_query(top_features, row=None):
    labels=[f.replace('_',' ') for f in top_features]
    base="kidney stone nephrolithiasis recurrence risk factors"
    if labels:
        base += " " + " ".join(labels)
    return base

def predict(name,*v):
    if not MODEL.exists(): return "Run training first","","","",history()
    vals=v[:-1]; llm=v[-1]
    row=pd.DataFrame([dict(zip(FEATURES,[
        vals[0],vals[1],vals[2],int(vals[3]),int(vals[4]),int(vals[5]),int(vals[6]),vals[7],
        vals[8],vals[9],vals[10],vals[11],int(vals[12]),int(vals[13]),vals[14],vals[15],
        vals[16],vals[17],vals[18],vals[19],vals[20],vals[21],vals[22],vals[23],vals[24],
        vals[25],vals[26],vals[27]]))])
    m=joblib.load(MODEL)
    p=int(m.predict(row)[0]); prob=float(m.predict_proba(row)[0,1])
    res=("Higher predicted recurrence risk" if p else "Lower predicted recurrence risk")
    res+=f"\nEstimated model probability: {prob:.1%}"
    xai,top_features=local_explanation(row,m,prob)

    # Make retrieval query depend on the actual model explanation instead of a fixed query.
    query=build_rag_query(top_features,row)
    rag_results=rag.retrieve_with_scores(query,4)
    ev=format_evidence(rag_results)
    explanation=xai+"\n\n"+ev
    if llm:
        try:
            evidence_text="\n\n".join(
                f"[{r['title']}] (retrieval score {r['score']:.3f})\n{r['text']}"
                for r in rag_results
            )
            explanation=ollama(
                f"ML result: {res}\nXAI:\n{xai}\n"
                f"Retrieved evidence:\n{evidence_text}\n\n"
                "Explain the result for an academic audience in simple language. "
                "Use only the retrieved evidence. Do not diagnose, prescribe, or claim causation."
            ) + "\n\n### 🔎 Retrieved evidence used\n" + ev
        except Exception as e:
            explanation += f"\n\n⚠️ LLM unavailable: {e}\nThe ML prediction and retrieved evidence remain available."
    c=db()
    c.execute("insert into predictions(patient,prediction,probability,time) values(?,?,?,datetime('now'))",
              (name,p,prob));c.commit();c.close()
    return res,f"{prob:.1%}",explanation,"Prediction saved.",history()

with gr.Blocks(title="Nephrolithiasis Recurrence Prediction") as demo:
    gr.Markdown("""# 🩺 AI-Based Prediction for Nephrolithiasis Recurrence
### Final-Year AI/ML Research Prototype

**Machine Learning + Cross-Validation + XAI + RAG + Optional Local LLM**

>
""")
    with gr.Tabs():
        with gr.Tab("🔬 Prediction"):
            name=gr.Textbox(label="Patient Name")
            nums=[gr.Number(label=x,value=v) for x,v in [
                ("Age",40),("BMI",25),("Current Medications",0),("Daily Water Intake",2),
                ("Daily Citrus Intake",1),("Daily Salt Intake",6),("Serum Calcium",9.5),
                ("Urine pH",6.1),("Serum Uric Acid",6),("Serum iPTH",48),("Creatinine",1),
                ("Vitamin D",28),("24h Urine Calcium",190),("24h Urine Phosphorous",700),
                ("24h Urine Oxalate",35),("24h Urine Uric Acid",550),
                ("24h Urine Citrate",600),("Stone Burden",5)]]
            age,bmi,meds,water,citrus,salt,sca,ph,sua,ipth,creat,vitd,uca,uph,ox,uu,ci,burden=nums
            gender=gr.Radio(["Male","Female"],value="Male",label="Gender")
            ht=gr.Checkbox(label="Hypertension");diab=gr.Checkbox(label="Diabetes")
            prev=gr.Checkbox(label="Previous Stones");fam=gr.Checkbox(label="Family History")
            alcohol=gr.Dropdown(["None","Moderate","High"],value="None",label="Alcohol")
            calcium=gr.Checkbox(label="Normal Calcium Diet");potassium=gr.Checkbox(label="High Potassium Diet")
            treat=gr.Dropdown(["Medical","Surgical","Observation"],value="Medical",label="Treatment")
            surg=gr.Dropdown(["None","PCNL","URS","ESWL","Open"],value="None",label="Surgery")
            llm=gr.Checkbox(label="Use local Ollama LLM (optional)")
            btn=gr.Button("🔍 Predict Recurrence Risk",variant="primary")
            with gr.Row():
                res=gr.Textbox(label="Prediction",lines=2);prob=gr.Textbox(label="Risk Probability")
            status=gr.Textbox(label="Status")
            ex=gr.Markdown()
            hist=gr.Dataframe(label="Recent Prediction History",interactive=False)
            btn.click(predict,[name,age,gender,bmi,ht,diab,prev,fam,alcohol,meds,water,citrus,salt,
                              calcium,potassium,sca,ph,sua,ipth,creat,vitd,uca,uph,ox,uu,ci,treat,surg,burden,llm],
                      [res,prob,ex,status,hist])

        with gr.Tab("📊 Analytics Dashboard"):
            gr.Markdown("# 📊 Model Performance Dashboard")
            gr.Markdown("Graphs are generated from the trained model and evaluation artifacts.")
            card=gr.Markdown(cards)
            refresh=gr.Button("🔄 Refresh Dashboard")
            with gr.Row():
                a=gr.Plot(accplot); b=gr.Plot(multplot)
            with gr.Row():
                c=gr.Plot(cvplot); d=gr.Plot(fiplot)
            with gr.Row():
                e=gr.Plot(cmplot); f=gr.Plot(lambda:curveplot("roc"))
            with gr.Row():
                g=gr.Plot(lambda:curveplot("pr"))
            refresh.click(lambda:(cards(),accplot(),multplot(),cvplot(),fiplot(),cmplot(),
                                  curveplot("roc"),curveplot("pr")),
                          outputs=[card,a,b,c,d,e,f,g])

        with gr.Tab("📈 EDA"):
            gr.Markdown("""# 📈 Exploratory Data Analysis
These plots show the structure and quality of the dataset used by the training pipeline.""")
            with gr.Row():
                gr.Plot(eda_class);gr.Plot(eda_missing)
            gr.Plot(eda_dist)
            gr.Markdown("### Data Quality")
            def quality():
                d=pd.read_csv(DATA)
                return pd.DataFrame({
                    "rows":[len(d)],"features":[len(FEATURES)],
                    "missing_cells":[int(d[FEATURES].isna().sum().sum())],
                    "recurrence_rate":[f"{d.recurrence.mean():.2%}"]
                })
            gr.Dataframe(value=quality,interactive=False)

        with gr.Tab("📋 Model Results"):
            gr.Markdown("### Hold-out Test Results")
            gr.Dataframe(value=metrics,interactive=False)
            gr.Markdown("### 5-Fold Cross-Validation Results")
            gr.Dataframe(value=lambda:pd.read_csv(CVALL) if CVALL.exists() else pd.DataFrame(),interactive=False)
            gr.Markdown("### Hyperparameter Tuning")
            gr.Dataframe(value=lambda:pd.read_csv(ART/"hyperparameter_tuning.csv") if (ART/"hyperparameter_tuning.csv").exists() else pd.DataFrame(),interactive=False)

        with gr.Tab("🧠 XAI + RAG"):
            gr.Markdown("""# 🧠 Explainable AI + Retrieval-Augmented Generation

**XAI** explains which input features pushed the model score up or down.

**RAG** retrieves and ranks supporting educational material from the local knowledge base. Each result shows its section title and retrieval score so the process is transparent.

**Optional LLM** can turn the retrieved evidence into a simpler explanation. The LLM does not make the prediction.

**Pipeline:** ML prediction → local XAI → feature-aware RAG query → ranked evidence → optional LLM summary.
""")
            q=gr.Textbox(label="Ask the knowledge base",
                         value="What factors are associated with kidney stone recurrence?")
            ask=gr.Button("🔎 Retrieve Ranked Evidence",variant="primary")
            ans=gr.Markdown()
            def retrieve_for_ui(query):
                return format_evidence(rag.retrieve_with_scores(query,4))
            ask.click(retrieve_for_ui,q,ans)

if __name__=="__main__":
    demo.launch()
