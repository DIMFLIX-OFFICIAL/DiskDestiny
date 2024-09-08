import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_USE = os.environ.get('WEIGHTS_USE', 'true').lower() == 'true'
WEIGHTS = {
	key.replace("_WEIGHT", ""): value 
	for key, value in os.environ.items() if key.endswith('_WEIGHT')
}

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_SRC = Path(__file__).parent


PATH_TO_DATA = PROJECT_ROOT / "data"
PATH_TO_PARAMS = PROJECT_SRC / "params"

PATH_TO_PREDICT_DATASETS = PROJECT_ROOT / "data" / "prediction"
PATH_TO_PREDICT_NORMALIZED_DATASETS = PROJECT_ROOT / "data" / "prediction" / "normalized"
PATH_TO_TRAIN_DATASETS = PROJECT_ROOT / "data" / "train"

PATH_TO_WEIGHTS = PROJECT_SRC / "models" / "weights"
PATH_TO_COLUMNS = PATH_TO_PARAMS / "column_good.json"
PATH_TO_MEDIANS = PATH_TO_PARAMS / "medians_value.json"
PATH_TO_MODELS_CODING = PATH_TO_PARAMS / "model_coding.json"
PATH_TO_START_HDD = PATH_TO_PARAMS / "start_date_HDD.csv"
