"""Database operations module."""
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
from pymongo import MongoClient
from pymongo.collection import Collection
from . import db_config
import pandas as pd

class DatabaseOperations:
    def __init__(self):
        self.client = MongoClient(db_config.MONGODB_URI)
        self.db = self.client[db_config.DB_NAME]
        
    def _get_collection(self, collection_name: str) -> Collection:
        return self.db[collection_name]
        
    def insert_patient(self, patient_data: Dict[str, Any]) -> str:
        """Insert a new patient record"""
        collection = self._get_collection(db_config.PATIENT_COLLECTION)
        patient_data['created_at'] = datetime.now(UTC)
        result = collection.insert_one(patient_data)
        return str(result.inserted_id)
        
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient information by ID"""
        collection = self._get_collection(db_config.PATIENT_COLLECTION)
        return collection.find_one({'patient_id': patient_id})
        
    def update_patient(self, patient_id: str, update_data: Dict[str, Any]) -> bool:
        """Update patient information"""
        collection = self._get_collection(db_config.PATIENT_COLLECTION)
        update_data['updated_at'] = datetime.now(UTC)
        result = collection.update_one(
            {'patient_id': patient_id},
            {'$set': update_data}
        )
        return result.modified_count > 0
        
    def insert_biomarker_data(self, data: Dict[str, Any]) -> str:
        """Insert new biomarker data"""
        collection = self._get_collection(db_config.BIOMARKER_COLLECTION)
        data['recorded_at'] = datetime.now(UTC)
        result = collection.insert_one(data)
        return str(result.inserted_id)
        
    def get_patient_biomarkers(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get biomarker data for a patient"""
        collection = self._get_collection(db_config.BIOMARKER_COLLECTION)
        cursor = collection.find({'patient_id': patient_id}).sort('recorded_at', -1)
        return list(cursor)
        
    def insert_regional_data(self, data: Dict[str, Any]) -> str:
        """Insert new regional health data"""
        collection = self._get_collection(db_config.REGIONAL_COLLECTION)
        data['recorded_at'] = datetime.now(UTC)
        result = collection.insert_one(data)
        return str(result.inserted_id)
        
    def get_regional_stats(self, region: str) -> Optional[Dict[str, Any]]:
        """Get current health statistics for a region"""
        collection = self._get_collection(db_config.REGIONAL_COLLECTION)
        return collection.find_one({'region': region}, sort=[('recorded_at', -1)])
        
    def get_regional_trends(self, region: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get regional trend data within a date range"""
        collection = self._get_collection(db_config.REGIONAL_COLLECTION)
        cursor = collection.find({
            'region': region,
            'recorded_at': {'$gte': start_date, '$lte': end_date}
        }).sort('recorded_at', 1)
        return list(cursor)
    
    def load_initial_data(self):
        """Load initial data from CSV files into the database"""
        # Load patient data
        patients_df = pd.read_csv('data/patient_data.csv')
        for _, row in patients_df.iterrows():
            self.insert_patient(row.to_dict())
            
        # Load biomarker data
        biomarkers_df = pd.read_csv('data/biomarker_data.csv')
        for _, row in biomarkers_df.iterrows():
            self.insert_biomarker_data(row.to_dict())
            
        # Load regional data
        regional_df = pd.read_csv('data/regional_data.csv')
        for _, row in regional_df.iterrows():
            self.insert_regional_data(row.to_dict())

# Create a global instance
db_ops = DatabaseOperations()