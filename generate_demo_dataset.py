from pathlib import Path
import numpy as np
import pandas as pd

# DEMONSTRATION ONLY.
# The label is generated from a transparent synthetic rule so that the complete
# software pipeline can be demonstrated. It is not clinical data.

rng = np.random.default_rng(42)
N = 3000

df = pd.DataFrame({
    "age": rng.integers(18, 80, N),
    "gender": rng.choice(["Male","Female"], N),
    "bmi": np.clip(rng.normal(25, 4.2, N), 16, 44),
    "hypertension": rng.binomial(1,.25,N),
    "diabetes": rng.binomial(1,.18,N),
    "previous_stones": rng.binomial(1,.38,N),
    "family_history": rng.binomial(1,.28,N),
    "alcohol_consumption": rng.choice(["None","Moderate","High"],N,p=[.55,.35,.10]),
    "current_medications": rng.poisson(1.5,N),
    "daily_water_intake": np.clip(rng.normal(2.1,.55,N),.5,5),
    "daily_citrus_intake": np.clip(rng.normal(1.2,.6,N),0,5),
    "daily_salt_intake": np.clip(rng.normal(6.2,1.8,N),1,15),
    "normal_calcium_diet": rng.binomial(1,.68,N),
    "high_potassium_diet": rng.binomial(1,.55,N),
    "serum_calcium": np.clip(rng.normal(9.5,.4,N),7.5,12),
    "urine_ph": np.clip(rng.normal(6.1,.65,N),4.5,8.5),
    "serum_uric_acid": np.clip(rng.normal(6,1.1,N),2,11),
    "serum_ipth": np.clip(rng.normal(48,16,N),10,150),
    "serum_creatinine": np.clip(rng.normal(1,.2,N),.4,3),
    "serum_vitamin_d": np.clip(rng.normal(28,9,N),5,70),
    "urine_24h_calcium": np.clip(rng.normal(190,70,N),20,600),
    "urine_24h_phosphorous": np.clip(rng.normal(700,200,N),100,1500),
    "urine_24h_oxalate": np.clip(rng.normal(35,11,N),5,120),
    "urine_24h_uric_acid": np.clip(rng.normal(550,160,N),100,1200),
    "urine_24h_citrate": np.clip(rng.normal(600,220,N),50,1600),
    "renal_calculi_treatment": rng.choice(["Medical","Surgical","Observation"],N,p=[.45,.4,.15]),
    "surgery_type": rng.choice(["None","PCNL","URS","ESWL","Open"],N,p=[.45,.15,.2,.18,.02]),
    "stone_burden": np.clip(rng.gamma(2,3.5,N),.2,40),
})

# Strong but synthetic recurrence signal for demonstration.
score = (
    2.0*df.previous_stones
    + 1.2*df.family_history
    + .8*df.hypertension
    + .7*df.diabetes
    + .045*(df.stone_burden-5)
    + .04*(df.urine_24h_oxalate-35)
    - .9*(df.daily_water_intake-2)
    + .12*(df.daily_salt_intake-6)
    - .002*(df.urine_24h_citrate-600)
    + .02*(df.age-40)
)
# Threshold with a small amount of label noise to avoid a perfectly deterministic target.
threshold = np.median(score)
noise = rng.normal(0, .22, N)
df["recurrence"] = ((score + noise) > threshold).astype(int)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/nephrolithiasis.csv",index=False)
print("DEMONSTRATION dataset created:", df.shape)
print("This dataset is synthetic and must not be presented as clinical evidence.")
