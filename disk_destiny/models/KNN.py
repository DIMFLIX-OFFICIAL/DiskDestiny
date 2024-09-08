from pandas import DataFrame
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import pickle
from typing import List

from .base import BaseModel
from schemes import TrainData
from config import PATH_TO_WEIGHTS


class KNN(BaseModel):
	name = 'KNN'

	def __init__(self, weight: float = 1) -> None:
		super().__init__(weight=weight)
		self.model = KNeighborsClassifier()
	
	def _train(self,  data: TrainData) -> None:
		def get_matrix(model, data: TrainData) -> dict:
			pred = model.predict(data.X_test)
			metrics = {}
			for i, column in enumerate(data.y_test.columns):
				accuracy = accuracy_score(data.y_test[column], pred[:, i])
				f1 = f1_score(data.y_test[column], pred[:, i])

				metrics[column] = {
					'Accuracy':accuracy,
					'F1 Score': f1
				}
			return metrics

		with open(PATH_TO_WEIGHTS / "KNN.pkl", 'rb') as f:
			self.model = pickle.load(f)

		# Считаем метрики старой модели
		old_metrics = get_matrix(self.model, data)
		mean_old_f1 = np.mean([v['F1 Score'] for v in old_metrics.values()])
		mean_old_accuracy = np.mean([v['Accuracy'] for v in old_metrics.values()])
		
		# Обучение модели 
		param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan']}
		grid_search = GridSearchCV(self.model, param_grid, cv=5, verbose=2)
		grid_search.fit(data.X_train, data.y_train)
		self.model = grid_search.best_estimator_

		# Считаем метрики новой модели
		new_metrics = get_matrix(self.model, data)
		mean_new_f1 = np.mean([v['F1 Score'] for v in new_metrics.values()])
		mean_new_accuracy = np.mean([v['Accuracy'] for v in new_metrics.values()])

		# Сравниваем, стала-ли лучше модель или нет
		if mean_new_f1 > mean_old_f1 and mean_new_accuracy > mean_old_accuracy:
			with open(PATH_TO_WEIGHTS / "KNN.pkl", 'wb') as f:
				pickle.dump(self.model, f)
			print(f"Модель улучшилась после дообучения! Accuracy: {mean_old_accuracy} -> {mean_new_accuracy}, F1 Score: {mean_old_f1} -> {mean_new_f1}!")
		else:
			print("Модель не улучшилась после дообучения!")

	def _predict(self, X: DataFrame) -> List[list]:
		with open(PATH_TO_WEIGHTS / "KNN.pkl", 'rb') as f:
			self.model = pickle.load(f)
		
		predictions = self.model.predict(X[:10])
		return {key: value for key, value in zip(X.index.values, predictions)} 
