import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()

# Authentication
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER_TOKEN:
    print("WARNING: TWITTER_BEARER_TOKEN not found in environment.")

# Time Bounds for Study
START_DATE = pd.to_datetime("2020-03-25", utc=True)
END_DATE = pd.to_datetime("2020-09-15", utc=True)

# Hyperparameters
BERTWEET_MODEL = "vinai/bertweet-base"
BATCH_SIZE = 64
LSTM_WINDOW_SIZE = 14

# Keyword Lists
CRISIS_KEYWORDS = [
    "crisis", "emergency", "panic", "disaster", "outbreak",
    "lockdown", "quarantine", "fatal", "death", "severe"
]

RUMOR_KEYWORDS = [
    "hoax", "fake", "conspiracy", "5g", "bioweapon",
    "plandemic", "chip", "microchip", "cure", "miracle",
    "secret", "hidden"
]

# Paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
HYDRATED_DIR = os.path.join(DATA_DIR, "hydrated")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

EMBEDDINGS_DIR = "embeddings"
CHECKPOINTS_DIR = "checkpoints"
REPORTS_DIR = "reports"
LOGS_DIR = "logs"
