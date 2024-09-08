import inquirer
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Tuple

import models
from models.base import BaseModel
from utils.data_normalizing import Normalizer
from utils.other import print_pure_banner, get_csv_files, ask_list, ask_checkbox
from config import WEIGHTS, WEIGHTS_USE, PATH_TO_PREDICT_DATASETS, PATH_TO_PREDICT_NORMALIZED_DATASETS, PATH_TO_TRAIN_DATASETS


class Application:
	weights_use: bool
	allowed_models: Dict[str, BaseModel]
	
	def __init__(self, weights: Dict[str, float], weights_use: bool = True) -> None:
		self.weights_use = weights_use
		self.allowed_models = {}
		
		PATH_TO_PREDICT_DATASETS.mkdir(exist_ok=True, parents=True)
		PATH_TO_PREDICT_NORMALIZED_DATASETS.mkdir(exist_ok=True, parents=True)
		PATH_TO_TRAIN_DATASETS.mkdir(exist_ok=True, parents=True)

		for model in BaseModel.__subclasses__():
			weight =  weights.get(model.name.upper(), 1)

			try:
				weight = float(weight)
			except ValueError:
				raise ValueError(f"Вес для модели {model.name} должен быть числом")

			if weight < 0 or weight > 1:
				raise ValueError(f"Вес для модели {model.name} должен быть от 0 до 1")
			
			self.allowed_models[model.name] = model(weight=weight)
	
	def run(self) -> None:
		print_pure_banner()
		self.main_menu()
		
	def main_menu(self) -> None:
		main_menu_answer = ask_list("Что будем делать?", ['Дообучение модели', 'Получить предсказание', 'Выход'])
		print_pure_banner()

		match main_menu_answer:
			case 'Дообучение модели':
				self.train()

			case 'Получить предсказание':
				self.predict()

			case 'Выход':
				exit()
	
	def train(self) -> None:
		model = ask_list("Выберите модель для дообучения", self.allowed_models)
		print_pure_banner()

		csv_files = get_csv_files(path=PATH_TO_TRAIN_DATASETS)
		if not csv_files:
			print("Нет датасетов для обучения!")
			exit()

		dataset_filename = ask_list("Выберите тренировочный датасет.", csv_files)
		print(dataset_filename)


		data = Normalizer.get_df_for_train(PATH_TO_TRAIN_DATASETS / dataset_filename)
		self.allowed_models[model].train(data)

	def predict(self) -> float:
		list_models: List[BaseModel] = ask_checkbox("Выберите модели для предсказания.",self.allowed_models, default=list(self.allowed_models.keys()))
		list_models = [self.allowed_models[i] for i in list_models]
		print_pure_banner()

		csv_files = get_csv_files(path=PATH_TO_PREDICT_DATASETS)
		if not csv_files:
			print("Нет датасетов для предсказания!")
			exit()

		dataset_filename = ask_list("Выберите датасет.", csv_files)	

		##==> Нормализация данных
		##########################################################
		csv_path = PATH_TO_PREDICT_DATASETS/ dataset_filename
		csv_normalized_path = PATH_TO_PREDICT_NORMALIZED_DATASETS / dataset_filename

		if csv_normalized_path.exists() and inquirer.confirm("Данные уже преобразованы. Хотите продолжить без повторной обработки?", default=True):
			print("Продолжаем с преобразованными данными...")
			X = pd.read_csv(csv_normalized_path)
		else:
			print("Преобразуем данные... Это может занять некоторое время.")
			X = Normalizer.get_df_for_predict(csv_path)

		print_pure_banner()
		print("Начинаем предсказывание отказоустойчивости дисков...")
		results: Dict[float, np.ndarray] = {i.weight: i.predict(X=X) for i in list_models}

		##==> Подсчёт среднего отказа дисков по моделям
		##########################################################
		def calculate_destroyed_devices(shans: Dict[str, List[float]], count_models: int) -> Tuple[int, List[str]]:
			count_destroy = 0
			destroy_devices = []
			for key, values in shans.items():
				for month in range(4):
					if values[month] / count_models > 0.7:
						count_destroy += 1
						destroy_devices.append(key)
			return count_destroy, destroy_devices

		if self.weights_use:
			shans = {}
			count_models = len(results.keys())
			
			for weight, pred in results.items():
				for key in pred.keys():
					if key not in shans:
						shans[key] = [0, 0, 0, 0]
					for month in range(4):
						shans[key][month] += pred[key][month] * weight

			count_destroy_3month, destroy_devices3 = calculate_destroyed_devices(shans, count_models)
			count_destroy_6month, destroy_devices6 = calculate_destroyed_devices(shans, count_models)
			count_destroy_9month, destroy_devices9 = calculate_destroyed_devices(shans, count_models)
			count_destroy_12month, destroy_devices12 = calculate_destroyed_devices(shans, count_models)

		else:
			shans_no_weight = {}
			count_models = len(results.keys())
			
			for pred in results.values():
				for key in pred.keys():
					if key not in shans_no_weight:
						shans_no_weight[key] = [0, 0, 0, 0]
					for month in range(4):
						shans_no_weight[key][month] += pred[key][month]

			count_destroy_3month, destroy_devices3 = calculate_destroyed_devices(shans_no_weight, count_models)
			count_destroy_6month, destroy_devices6 = calculate_destroyed_devices(shans_no_weight, count_models)
			count_destroy_9month, destroy_devices9 = calculate_destroyed_devices(shans_no_weight, count_models)
			count_destroy_12month, destroy_devices12 = calculate_destroyed_devices(shans_no_weight, count_models)

		print(f"\n\n[{count_destroy_3month}] Устройств выйдет из строя в течение 3 месяцев: {destroy_devices3}")
		print(f"[{count_destroy_6month}] Устройств выйдет из строя в течение 6 месяцев: {destroy_devices6}")
		print(f"[{count_destroy_9month}] Устройств выйдет из строя в течение 9 месяцев: {destroy_devices9}")
		print(f"[{count_destroy_12month}] Устройств выйдет из строя в течение 12 месяцев: {destroy_devices12}")


if __name__ == "__main__":
	load_dotenv()
	app = Application(weights=WEIGHTS, weights_use=WEIGHTS_USE)
	app.run()
