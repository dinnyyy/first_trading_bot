import alpaca_trade_api as tradeapi
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta # Or your TA library
import time
from datetime import datetime, timedelta
import os # For API keys
from dotenv import load_dotenv # 1. Import the load_dotenv function


load_dotenv()

API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = os.environ.get('APCA_PAPER_URL')

