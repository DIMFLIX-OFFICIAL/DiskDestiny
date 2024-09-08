import json
import pandas as pd
from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
	PATH_TO_COLUMNS, 
	PATH_TO_MEDIANS, 
	PATH_TO_MODELS_CODING, 
	PATH_TO_START_HDD,
	PATH_TO_PREDICT_NORMALIZED_DATASETS
)
from schemes import TrainData


class Normalizer:

	@staticmethod
	def load_json(path: Union[str, Path]) -> dict:
		with open(path) as f:
			return json.load(f)
		
	@classmethod
	def get_df_for_predict(cls, csv: Path) -> pd.DataFrame:
		print("Получаем данные для предсказания...")

		columns_good = cls.load_json(PATH_TO_COLUMNS)["columns"]
		model_dict = cls.load_json(PATH_TO_MODELS_CODING)
		medians = cls.load_json(PATH_TO_MEDIANS)

		data1 = pd.read_csv(csv)
		data2 = pd.read_csv(PATH_TO_START_HDD)

		data = pd.merge(data1, data2, left_on='serial_number', right_on='serial_number_end', how='left')
		del data1, data2
		
		data['date'] = pd.to_datetime(data['date'])
		data['min_date'] = pd.to_datetime(data['min_date'])
		data['work_days'] = (data['date'] - data['min_date']).dt.days
		
		data['model'] = data['model'].map(model_dict)
		
		for col in columns_good[4:-4]:
			data[col] = data[col].fillna(medians[col+"_avg"])

		data.set_index('serial_number', inplace=True)
		data = data[columns_good[:-4]]
		data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

		PATH_TO_PREDICT_NORMALIZED_DATASETS.mkdir(exist_ok=True)
		data.to_csv(PATH_TO_PREDICT_NORMALIZED_DATASETS/csv.name, index=False)
		return data
	
	@classmethod
	def get_df_for_train(cls, csv: Path) -> TrainData:
		print("Получаем данные для предсказания...")
		
		columns_good = cls.load_json(PATH_TO_COLUMNS)["columns"]
		model_dict = cls.load_json(PATH_TO_MODELS_CODING)
			
		data = pd.read_csv(csv)
		data.set_index('serial_number', inplace=True)

		# Кодирование моделей, если это по каки-то причинам не произошло		
		for col, coding_dict in [('model', model_dict)]:
			data[col] = data[col].map(coding_dict)

		# Замена пропусков на 0, если это по каким-то причинам не произошло
		data = data[columns_good].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

		Y = data[['3_month', '6_month', '9_month', '12_month']]
		X = data.drop(['3_month', '6_month', '9_month', '12_month'], axis=1)
		

		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
		X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

		return TrainData(
			X_train, X_valid, X_test, 
			y_train, y_valid, y_test
		)
	