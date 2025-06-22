# Libyan Dialect Sentiment Analysis (Misurata Sub-dialect)

A robust, production-ready machine learning pipeline for sentiment analysis on Libyan Arabic text, specifically optimized for the Misurata sub-dialect. This project implements multiple machine learning models with hyperparameter tuning and comprehensive evaluation metrics.

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Results](#-results)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Multiple ML Models**:
  - Support Vector Machines (SVM) with Linear and RBF kernels
  - Naive Bayes
  - Logistic Regression
  
- **Advanced Features**:
  - TF-IDF with n-grams (1, 2, 3)
  - Custom Arabic text tokenization
  - Hyperparameter tuning with GridSearchCV
  
- **Comprehensive Evaluation**:
  - Detailed classification reports (CSV & HTML)
  - Confusion matrix visualization
  - ROC-AUC and Precision-Recall curves
  - F1-score per class analysis
  
- **Production Ready**:
  - Containerized with Docker
  - Environment-based configuration
  - Structured logging with color-coded output
  - Model persistence

## 🚀 Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.11+ (for local development)
- Git

## 🛠 Getting Started

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/m-elzalouk/libyan-dialect-sentiment.git
   cd libyan-dialect-sentiment
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Docker Setup (Recommended)

1. **Build and run the container**:
   ```bash
   # Start the training service in detached mode (work in background)
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   ```

2. **Stop the service**:
   ```bash
   docker-compose down
   ```

3. **Monitor resources**:
   ```bash
   # View running containers
   docker-compose ps
   
   # View resource usage
   docker stats
   ```

## 📁 Project Structure

```
libyan-dialect-sentiment/
│
├── data/                   # Dataset directory (mounted volume)
│   └── dataset_cleaned-positive-negative-v2.csv
│
├── scripts/               # Main application code
│   ├── train.py           # Training pipeline entry point
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── features.py        # Feature extraction
│   ├── models.py          # Model definitions and training
│   ├── evaluate.py        # Model evaluation and visualization
│   └── utils.py           # Utility functions and logging
│
├── models/               # Saved models (mounted volume)
├── results/              # Output results (mounted volume)
├── logs/                 # Log files (mounted volume)
│
├── .env.example         # Example environment variables
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Docker ignore file
└── README.md            # This file
```

## ⚙️ Configuration

Configure the application using the `.env` file. Copy from the example:

```bash
cp .env.example .env
```

Key configuration options:
- `DATA_PATH`: Path to the dataset file
- `RESULTS_DIR`: Directory to save evaluation results
- `MODELS_DIR`: Directory to save trained models
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `TEST_SIZE`: Fraction of data to use for testing (default: 0.2)
- `RANDOM_STATE`: Random seed for reproducibility

## 🚦 Usage

### Running the Pipeline

1. **Using Docker Compose (Recommended)**:
   ```bash
   # Start the training pipeline
   docker-compose up -d
   
   # Follow the logs
   docker-compose logs -f
   ```

2. **Using Python directly**:
   ```bash
   python -m scripts.train
   ```

### Viewing Results

After training completes, check the following directories:
- `results/`: Contains evaluation metrics and visualizations
- `models/`: Contains the trained model files
- `logs/`: Contains log files with timestamps

## 🧠 Model Training

The pipeline trains multiple models with hyperparameter tuning:

1. **Data Loading**: Loads and preprocesses the dataset
2. **Feature Extraction**: Converts text to TF-IDF features
3. **Model Training**:
   - SVM with Linear Kernel
   - SVM with RBF Kernel
   - Naive Bayes
   - Logistic Regression
4. **Hyperparameter Tuning**: Uses GridSearchCV for optimal parameters
5. **Evaluation**: Generates comprehensive metrics and visualizations

## 📊 Results

### Output Files

- `results/classification_report.html`: Interactive HTML report
- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/roc_curve.png`: ROC curve plot
- `results/scores.json`: Detailed metrics in JSON format
- `results/f1_scores.png`: F1-score comparison across classes

### Example Output

```
SUCCESS 2025-06-22 16:32:13 - models - Best model: SVM (Linear)
SUCCESS 2025-06-22 16:32:13 - models - Best F1_WEIGHTED: 0.923
INFO    2025-06-22 16:32:13 - evaluate - Detailed classification reports saved to results/classification_report.csv and results/classification_report.html
```

## 🛠 Development

### Setting Up for Development

1. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**:
   ```bash
   # Run unit tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=scripts tests/
   ```

3. **Code quality checks**:
   ```bash
   # Linting
   flake8 scripts/
   
   # Type checking
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

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
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
