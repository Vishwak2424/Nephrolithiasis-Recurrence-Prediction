NEPHROLITHIASIS RECURRENCE - ENHANCED FINAL YEAR PROTOTYPE

1. Use Python 3.12.
2. Open CMD in this folder.
3. Run: py -3.12 -m venv .venv
4. Run: .venv\Scripts\activate
5. Run: pip install -r requirements.txt
6. Run: python generate_demo_dataset.py
7. Run: python train.py
8. Run: python app.py
9. Open http://127.0.0.1:7860

New modules:
- Model comparison
- Hyperparameter tuning
- 5-fold cross-validation for all models
- EDA dashboard
- ROC and Precision-Recall graphs
- Confusion matrix
- Global permutation feature importance
- Local XAI contribution explanation for logistic models
- RAG evidence retrieval
- Optional local Ollama explanation

IMPORTANT:
The included dataset is synthetic demonstration data. Its metrics must not be presented as clinical validation.


RAG/XAI upgrade:
- RAG now displays ranked retrieval results with section titles and TF-IDF similarity scores.
- Prediction-time RAG queries are generated from the strongest local XAI feature contributions instead of a fixed query.
- Optional Ollama is instructed to use only the retrieved evidence and clearly remain educational.
