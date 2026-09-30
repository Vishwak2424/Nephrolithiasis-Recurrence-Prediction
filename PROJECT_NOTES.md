# What is genuinely established vs demonstration

The supplied mini-project gave the 28 input fields, Gradio UI, MySQL persistence and a pre-trained
model.pkl, but not the training dataset/model-generation code.

The reference project demonstrated an 80/20 split and comparison of ML algorithms with accuracy,
precision, recall and F1.

Current package therefore has two modes:

A. DEMONSTRATION MODE
   - fully executable immediately
   - synthetic recurrence labels
   - designed to demonstrate the requested high-accuracy ML workflow
   - NOT clinical evidence

B. REAL-DATA MODE
   - replace data/nephrolithiasis.csv
   - run TRAIN_REAL_DATA.bat
   - metrics are then calculated from the supplied real dataset

Do not report synthetic numbers as patient-study results.

Research context:
- ReSKU recurrence study: 423 training patients and 172 independent validation patients; best AUC
  0.65 training and 0.64 validation.
- EHR + 24-hour urine study: 1,231 patients; best 2-year AUC 0.62 and 5-year AUC 0.63.
- Large 2026 EHR study: 154,876 patients; test AUROC 0.727.
- 2026 Bern registry study: 706 patients; held-out AUC about 0.71 +/- 0.03.

These are literature benchmarks, not this package's results.
