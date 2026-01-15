# Loan Default Prediction

End-to-end ML pipeline for car loan default prediction with hyperparameter tuning, MLflow experiment tracking, FastAPI REST API, and Docker deployment.

## Problem Statement

Car loan companies face significant losses due to loan defaults, leading to stricter policies and higher rejection rates that negatively impact business by potentially rejecting stable clients. This project develops a credit risk scoring model to assess borrowers' probability of defaulting on their first installment. The model uses credit history, financial indicators, and loan information to predict default risk, enabling better lending decisions.

## Tech Stack

- **Python 3.11+**
- **ML/Data**: scikit-learn, pandas, numpy, imbalanced-learn (SMOTE)
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Uvicorn, Pydantic
- **Deployment**: Docker, Docker Compose
- **Visualization**: matplotlib, seaborn

## Project Structure

```
loan-default-prediction/
├── api/                # FastAPI application
│   ├── __init__.py
│   └── main.py        # API endpoints
├── config/             # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/           # Raw data files (car_loan.csv)
│   └── processed/     # Processed data
├── src/               # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and prediction
│   └── utils/         # Utility functions (logging)
├── models/            # Saved models and artifacts
├── mlruns/            # MLflow tracking data
├── notebooks/         # Jupyter notebooks for exploration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Performance

Multiple models were evaluated using GridSearchCV for hyperparameter tuning:

| Model | Test Accuracy | Test Precision (Class 1) | Test Recall (Class 1) |
|-------|---------------|--------------------------|----------------------|
| Logistic Regression | 0.759 | 0.29 | 0.13 |
| Decision Tree (Tuned) | 0.768 | 0.33 | 0.07 |
| Random Forest (Tuned) | 0.773 | 0.36 | 0.06 |


### Key Findings

- Applied SMOTE to handle severe class imbalance
- Feature engineering improved model performance (ID verification scores, loan burden ratios, credit stability metrics)
- Hyperparameter tuning with GridSearchCV optimized model performance
- All models struggled with minority class recall, highlighting the challenge of predicting loan defaults

## Setup & Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/arbaazali872/loan_default_analysis
cd loan-default-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Data Setup

Download car_loan dataset from: https://drive.google.com/file/d/1q1kZYypePCXZF94tTEv0oS7LUqJHY0yn/view?usp=drive_link
Place your `car_loan.csv` file in the `data/raw/` directory.

## Usage

### Training Models

Train models with hyperparameter tuning and MLflow tracking:

```bash
python -m src.models.train
```

This will:
- Load and preprocess data
- Apply feature engineering
- Train Logistic Regression, Random Forest, and Decision Tree with GridSearchCV
- Log experiments to MLflow
- Save the best model to `models/`

Training time: 30-60 minutes depending on hardware.

### View MLflow Experiments

Start the MLflow UI to view experiment tracking, compare models, and analyze metrics:

```bash
mlflow ui
```

Access at: `http://localhost:5000`

The UI shows:
- All experiment runs with parameters and metrics
- Model comparison charts
- Logged artifacts and models
- Performance visualizations

### Run API Locally

Start the FastAPI server:

```bash
python api/main.py
```

Or using uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access:
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Docker Deployment

#### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop
docker-compose down
```

#### Using Docker Directly

```bash
# Build image
docker build -t loan-default-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  --name loan-default-api \
  loan-default-api
```

#### View Container Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f loan-default-api
```

## Configuration

Model and pipeline settings can be adjusted in `config/config.yaml`:

- Data paths and split ratios
- Preprocessing strategies
- SMOTE parameters
- Model hyperparameters
- MLflow settings

