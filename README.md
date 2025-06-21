# Libyan Dialect Sentiment Analysis (Misurata Sub-dialect)

A CLI-based, Dockerized machine learning pipeline for sentiment analysis on Libyan Arabic poetry/texts.

## Project Structure

```
libya-sentiment-project/
│
├── data/
│   └── dataset_cleaned-positive-negative-v2.csv
├── scripts/
│   ├── train.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── utils.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env
├── README.md
```

## Main Features
- Clean architecture, modular scripts
- TfidfVectorizer with ngrams (1,2,3), max_features=20000
- SVM, Naive Bayes, Logistic Regression (with GridSearchCV)
- Classification report, confusion matrix, ROC-AUC
- Results saved as CSV and JSON
- Optional SMOTE for class balancing
- Docker + Python `.venv` virtual environment

## Quickstart

1. **Build Docker image:**
   ```bash
   docker build -t libya-sentiment .
   ```
2. **Run the pipeline:**
   ```bash
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/results:/app/results \
     -v $(pwd)/models:/app/models \
     libya-sentiment
   ```

## Configuration
- Adjust hyperparameters and paths in `scripts/config.py`
- Set environment variables in `.env` (optional)

## Outputs
- Best trained model: `models/best_model.pkl`
- Metrics: `results/scores.json`
- Confusion matrix: `results/confusion_matrix.png`
- ROC curve (if applicable): `results/roc_curve.png`

## Requirements
- Docker
- (No Jupyter or manual venv activation needed)

---

**Contact:** [Your Name]
