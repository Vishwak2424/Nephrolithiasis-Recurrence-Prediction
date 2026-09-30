# Nephrolithiasis Recurrence Knowledge Base

## Safety
This is an academic decision-support prototype, not a diagnostic or treatment system.

## Recurrence
Kidney stone recurrence is multifactorial. Previous stones, metabolic findings, urine chemistry,
dietary factors and clinical history can contribute to recurrence risk.

## 24-hour urine
Commonly evaluated measurements include urine volume, calcium, oxalate, citrate, uric acid,
sodium, potassium and pH. Their interpretation requires clinical context.

## Machine learning interpretation
A model probability is an estimate learned from training data. It does not establish that recurrence
will definitely occur or definitely not occur.

## Evaluation
Accuracy should be considered together with precision, recall, specificity, F1 and ROC-AUC.
External validation is particularly important for medical prediction.

## Literature
Published recurrence studies have often reported moderate discrimination rather than near-perfect
performance. Therefore any >90% result must be checked carefully for data leakage, target definition,
class imbalance and independent validation.

## Educational explanation
The application can retrieve these points and optionally ask a local LLM to explain the model output.
