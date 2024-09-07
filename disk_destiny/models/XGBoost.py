from .base import BaseModel


class XGBoost(BaseModel):
	name = 'XGBoost'

	def train(self) -> None:
		pass

	def predict(self) -> float:
		pass
