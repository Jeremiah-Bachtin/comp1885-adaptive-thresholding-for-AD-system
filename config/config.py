import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/foundational/comp1884_train_scores.csv")
IF_MODEL_PATH = os.getenv("IF_MODEL_PATH", "models/lstm_ae_best.h5")
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_ae_best.h5")
WINDOW_IF = int(os.getenv("WINDOW_IF", 14))
WINDOW_AE = int(os.getenv("WINDOW_AE", 30))
THRESH_QUANT_IF = float(os.getenv("THRESH_QUANT_IF", 0.03))
THRESH_QUANT_AE = float(os.getenv("THRESH_QUANT_AE", 0.95))
