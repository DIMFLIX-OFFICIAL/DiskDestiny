import pandas as pd
from dataclasses import dataclass


@dataclass
class TrainData:
	X_train: pd.DataFrame
	X_valid: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.DataFrame
	y_valid: pd.DataFrame
	y_test: pd.DataFrame
