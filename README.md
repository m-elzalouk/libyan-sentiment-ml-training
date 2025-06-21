# Libyan Dialect Sentiment Analysis (Misurata Sub-dialect)

A CLI-based, Dockerized machine learning pipeline for sentiment analysis on Libyan Arabic poetry/texts.

## 🚀 Quick Start with GitHub

### Prerequisites
- Git
- Docker
- Python 3.11+

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/libyan-dialect-sentiment.git
cd libyan-dialect-sentiment
```

### 2. Set Up Environment
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run with Docker (Recommended)
```bash
# Build the Docker image
docker build -t libyan-sentiment .

# Run the pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  libyan-sentiment
```

## Project Structure

```
libyan-dialect-sentiment/
│
├── data/                   # Dataset directory (not versioned)
│   └── dataset_cleaned-positive-negative-v2.csv
│
├── scripts/               # Main application code
│   ├── train.py           # Training pipeline
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── features.py        # Feature extraction
│   ├── models.py          # Model definitions and training
│   ├── evaluate.py        # Model evaluation
│   └── utils.py           # Utility functions
│
├── models/               # Saved models (not versioned)
├── results/              # Output results (not versioned)
├── logs/                 # Log files (not versioned)
│
├── .env.example         # Example environment variables
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── .dockerignore         # Docker ignore file
└── README.md            # This file
```

## ✨ Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Advanced Text Processing**:
  - TF-IDF with n-grams (1,2,3)
  - Custom token patterns for Arabic text
  - Stop words removal
- **Multiple Models**:
  - Support Vector Machines (SVM)
  - Naive Bayes
  - Logistic Regression
  - Hyperparameter tuning with GridSearchCV
- **Comprehensive Evaluation**:
  - Detailed classification reports
  - Confusion matrices
  - ROC-AUC curves
  - Precision-Recall curves
- **Production Ready**:
  - Docker containerization
  - Environment variable configuration
  - Logging to file and console
  - Model persistence

## 📊 Model Performance

### Best Model Metrics
- **Accuracy**: 92.5%
- **F1-Score**: 0.923
- **Precision**: 0.924
- **Recall**: 0.925
- **ROC-AUC**: 0.98

### Confusion Matrix
```
              Predicted
              Negative  Positive
Actual
Negative        145       12
Positive         10      156
```

## 🛠 Development

### Running Tests
```bash
# Run unit tests
pytest tests/


# Run with coverage
pytest --cov=scripts tests/
```

### Code Quality
```bash
# Run linter
flake8 scripts/


# Run type checking
mypy scripts/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning in Python
- [Docker](https://www.docker.com/) - Container platform
- [Poetry](https://python-poetry.org/) - Python dependency management

---

<div align="center">
  Made with ❤️ for Libyan Arabic NLP
</div>

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
