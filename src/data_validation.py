from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, validator
from enum import Enum

class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"

class AgeGroup(str, Enum):
    CHILD = "0-17"
    YOUNG_ADULT = "18-35"
    ADULT = "36-55"
    SENIOR = "56+"

class PatientData(BaseModel):
    patient_id: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    visit_date: datetime
    diagnosis: str = Field(..., min_length=1)
    medications: List[str] = Field(default_factory=list)
    
    @validator('medications')
    def validate_medications(cls, v):
        if not all(isinstance(med, str) and med.strip() for med in v):
            raise ValueError("All medications must be non-empty strings")
        return v

class BiomarkerData(BaseModel):
    patient_id: str = Field(..., min_length=1)
    biomarker_name: str = Field(..., min_length=1)
    value: float = Field(..., ge=0)
    recorded_at: datetime
    reference_range: Dict[str, float] = Field(default_factory=dict)
    
    @validator('reference_range')
    def validate_reference_range(cls, v):
        if v and ('min' not in v or 'max' not in v):
            raise ValueError("Reference range must include 'min' and 'max' values")
        if v and v['min'] >= v['max']:
            raise ValueError("Reference range min must be less than max")
        return v

class RegionalData(BaseModel):
    region: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    health_index: float = Field(..., ge=0, le=100)
    patient_count: int = Field(..., ge=0)
    age_group: AgeGroup
    gender: Gender
    diagnosis: str = Field(..., min_length=1)

def validate_patient_data(data: Dict[str, Any]) -> PatientData:
    """Validate patient data against schema"""
    return PatientData(**data)

def validate_biomarker_data(data: Dict[str, Any]) -> BiomarkerData:
    """Validate biomarker data against schema"""
    return BiomarkerData(**data)

def validate_regional_data(data: Dict[str, Any]) -> RegionalData:
    """Validate regional data against schema"""
    return RegionalData(**data)

def validate_dataframe(df: pd.DataFrame, schema: type) -> bool:
    """Validate pandas DataFrame against a schema"""
    try:
        for _, row in df.iterrows():
            schema(**row.to_dict())
        return True
    except Exception as e:
        raise ValueError(f"DataFrame validation failed: {str(e)}")

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    def __init__(self, message: str, errors: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.errors = errors or [] 