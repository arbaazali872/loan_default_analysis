from .train import main as train_main
from .predict import predict, predict_single

__all__ = ['train_main', 'predict', 'predict_single']