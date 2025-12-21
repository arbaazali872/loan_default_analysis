from setuptools import setup, find_packages

setup(
    name="loan_default_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas==2.1.4",
        "numpy==1.26.2",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "imbalanced-learn==0.11.0",
        "mlflow==2.9.2",
        "pyyaml==6.0.1",
        "joblib==1.3.2",
    ],
    python_requires=">=3.8",
)