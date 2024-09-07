from abc import ABC, abstractmethod


class BaseModel(ABC):
	name: str
	weight: float
	
	def __init__(self, weight: float = 1) -> None:
		self.weight = weight

	@abstractmethod
	def train(self) -> None: ...

	@abstractmethod
	def predict(self) -> float: ...
