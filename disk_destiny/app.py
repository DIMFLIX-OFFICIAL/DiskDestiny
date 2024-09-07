import os
import inquirer
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List

import models
from models.base import BaseModel
from utils.other import print_pure_banner


class App:
	allowed_models: Dict[str, BaseModel]
	
	def __init__(self, weights: Dict[str, float], weights_use: bool = True) -> None:
		self.allowed_models = {}

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
		self.ask_questions()
		
	def ask_questions(self) -> None:
		main_menu_answer = inquirer.prompt([
			inquirer.List('answer',
				message="Что будем делать?",
				choices=['Дообучение модели', 'Получить предсказание', 'Выход'],
			)
		])['answer']

		print_pure_banner()

		match main_menu_answer:
			case 'Дообучение модели':
				model = inquirer.prompt([
					inquirer.List('answer',
						message="Выберите модель для дообучения",
						choices=self.allowed_models,
					)
				])['answer']
				self.allowed_models[model].train()

			case 'Получить предсказание':
				models = inquirer.prompt([
					inquirer.Checkbox('answer',
						message="Выберите модели для предсказания.",
						choices=self.allowed_models,
					)
				])['answer']
				print(models)
				self.predict(list_models=models)

			case 'Выход':
				exit()
	
	def predict(self, list_models: List[BaseModel], use_weights: bool = True) -> float:
		results = {i.weight: i.predict() for i in list_models}
		
		if use_weights:
			return np.average(results.values(), axis=0, weights=results.keys())
		else:
			return np.mean(results.values(), axis=0)


if __name__ == "__main__":
	load_dotenv()
	weights_use = os.environ.get('WEIGHTS_USE', 'true').lower() == 'true'

	weights = {
		key.replace("_WEIGHT", ""): value 
		for key, value in os.environ.items() if key.endswith('_WEIGHT')
	}

	app = App(weights=weights, weights_use=weights_use)
	app.run()
