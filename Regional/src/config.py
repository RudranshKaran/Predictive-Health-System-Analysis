from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

MONGODB_URI = os.getenv('MONGODB_URI')
REGIONS = ["North", "South", "East", "West", "Central"]
DEFAULT_ANALYSIS_DAYS = 30