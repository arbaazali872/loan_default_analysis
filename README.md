# Loan Default Prediction

End-to-end ML pipeline for car loan default prediction with hyperparameter tuning, MLflow experiment tracking, Flask API, Gradio UI, and Docker deployment.

## Problem Statement

Car loan companies face significant losses due to loan defaults, leading to stricter policies and higher rejection rates that negatively impact business by potentially rejecting stable clients. This project develops a credit risk scoring model to assess borrowers' probability of defaulting on their first installment. The model leverages credit history, financial indicators, and loan information to predict default risk, enabling better lending decisions.

## Project Challenges & Key Learnings

### Challenges Faced
1. **Severe Class Imbalance**: Dataset had only 21.7% defaults, causing models to favor majority class
2. **High RAM Usage**: Training on 233k rows consumed 75% of 16GB RAM, causing system slowdowns
3. **Poor Minority Class Recall**: Initial models had ~64% recall, missing too many defaults
4. **XGBoost Overfitting**: With aggressive threshold tuning, XGBoost predicted everything as default (recall 1.0, precision 0.0)

### Solutions Implemented
1. **Stratified Sampling**: Reduced dataset to 30% (70k rows) while maintaining class distribution
   - Result: 70% less RAM, 3x faster training, no performance loss
2. **SMOTE Oversampling**: Applied 0.3 ratio to balance minority class during training
3. **Threshold Tuning**: Lowered prediction threshold from 0.5 to 0.35 to prioritize recall
4. **Model Selection**: Logistic Regression outperformed XGBoost and Random Forest despite lower complexity

### Final Outcome
- **Best Model**: Logistic Regression with 96.4% recall, 22.9% precision
- **Key Insight**: For loan defaults, missing a bad loan (false negative) is more costly than rejecting a good loan (false positive)
- **Trade-off**: Accepted lower precision to achieve high recall, as business context demands catching defaults

## Tech Stack

- **Python 3.11+**
- **ML/Data**: scikit-learn, pandas, numpy, imbalanced-learn (SMOTE), XGBoost
- **Experiment Tracking**: MLflow
- **API**: Flask
- **UI**: Gradio
- **Deployment**: Docker, Docker Compose
- **Visualization**: matplotlib, seaborn 

## Project Structure

```
loan-default-prediction/
├── app/                # Flask API and Gradio UI
│   ├── __init__.py
│   ├── flask_app.py   # Flask API endpoints
│   └── gradio_app.py  # Gradio web interface
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

Final results after stratified sampling (30%), SMOTE (0.3), and threshold tuning (0.35):

| Model | Test Accuracy | Test Precision (Class 1) | Test Recall (Class 1) | Test ROC-AUC |
|-------|---------------|--------------------------|----------------------|--------------|
| Logistic Regression ✓ | 0.323 | 0.235 | 0.940 | 0.630 |
| Random Forest | 0.335 | 0.237 | 0.929 | 0.631 |
| XGBoost | 0.217 | 0.217 | 1.000 | 0.617 |

**Selected Model**: Logistic Regression - Best balance between recall and precision

### Key Findings

- Stratified sampling (30%) reduced training time and RAM usage without sacrificing performance
- SMOTE at 0.3 ratio provided optimal minority class representation
- Lowering prediction threshold to 0.35 significantly improved recall
- Feature engineering (ID verification scores, loan burden ratios, credit stability) improved model performance
- XGBoost collapsed into a degenerate model, predicting all samples as default due to combined effect of scale_pos_weight, SMOTE, and low threshold. Recall 1.0 with precision equal to the dataset's default rate (21.7%)
- Pure recall maximization in model selection risks picking degenerate models; a minimum precision threshold of 0.22 was added to filter them out
- Logistic Regression proved most robust for this imbalanced classification task

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

Place your `car_loan.csv` file in the `data/raw/` directory.

## Usage

### Training Models

Train models with hyperparameter tuning and MLflow tracking:

```bash
python -m src.models.train
```

This will:
- Load and preprocess data with 30% stratified sampling
- Apply feature engineering
- Train Logistic Regression, Random Forest, and XGBoost with GridSearchCV
- Log experiments to MLflow
- Save the best model (highest recall) to `models/`

Training time: 15-30 minutes with 30% sampling.

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

### Run Flask API

Start the Flask API server:

```bash
python app/flask_app.py
```

API runs at: `http://localhost:5001`

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Predict loan default

### Run Gradio UI

Start the Gradio web interface (Flask API must be running first):

```bash
python app/gradio_app.py
```

Access at: `http://localhost:7860`

The Gradio interface provides:
- Form-based input for all loan application features
- Real-time prediction via Flask API
- Probability display for both classes (No Default / Default)

### Docker Deployment

#### Using Docker Compose

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
  -p 5000:5000 \
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
- Stratified sampling size (default: 0.3)
- Preprocessing strategies
- SMOTE parameters (sampling_strategy: 0.3)
- Prediction threshold (default: 0.35)
- Model hyperparameters
- MLflow settings
