from abc import ABC, abstractmethod
from pandas import DataFrame

from schemes import TrainData


class BaseModel(ABC):
	name: str
	weight: float
	
	def __init__(self, weight: float = 1) -> None:
		self.weight = weight

	def train(self, data: TrainData) -> None:
		print(f"Начинаем обучение модели {self.name}")
		return self._train(data)

	@abstractmethod
	def _train(self,  data: TrainData) -> None: ...


	def predict(self, X: DataFrame) -> list: 
		print(f"[{self.name}] Модель предсказывает результат...")
		result = self._predict(X)
		return result

	@abstractmethod
	def _predict(self, X: DataFrame) -> list: ...
