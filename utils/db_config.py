"""Database configuration module."""
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = 'health_system'

# Collections
PATIENT_COLLECTION = 'patients'
BIOMARKER_COLLECTION = 'biomarkers'
REGIONAL_COLLECTION = 'regional_data'