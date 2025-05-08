import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_regional_data(num_entries=100):
    # Define regions and their approximate coordinates
    regions = {
        'North': {'lat_range': (34.0, 36.0), 'lon_range': (-78.0, -76.0)},
        'South': {'lat_range': (32.0, 34.0), 'lon_range': (-78.0, -76.0)},
        'East': {'lat_range': (33.0, 35.0), 'lon_range': (-76.0, -74.0)},
        'West': {'lat_range': (33.0, 35.0), 'lon_range': (-80.0, -78.0)},
        'Central': {'lat_range': (33.0, 35.0), 'lon_range': (-77.0, -75.0)}
    }
    
    # Common health conditions
    conditions = [
        'Hypertension', 'Diabetes Type 2', 'Obesity', 
        'Respiratory Infection', 'Arthritis', 'Anxiety',
        'Depression', 'Asthma', 'Heart Disease', 'Allergies'
    ]
    
    # Generate data
    data = []
    for _ in range(num_entries):
        region = random.choice(list(regions.keys()))
        region_coords = regions[region]
        
        entry = {
            'patient_id': f'P{random.randint(1000, 9999)}',
            'region': region,
            'latitude': random.uniform(region_coords['lat_range'][0], region_coords['lat_range'][1]),
            'longitude': random.uniform(region_coords['lon_range'][0], region_coords['lon_range'][1]),
            'age': random.randint(18, 85),
            'gender': random.choice(['M', 'F']),
            'diagnosis': random.choice(conditions),
            'bmi': round(random.uniform(18.5, 35.0), 1),
            'blood_pressure_systolic': random.randint(90, 180),
            'blood_pressure_diastolic': random.randint(60, 110),
            'heart_rate': random.randint(60, 100),
            'cholesterol': random.randint(150, 300),
            'blood_sugar': random.randint(70, 200),
        }
        
        # Add age group
        if entry['age'] < 30:
            entry['age_group'] = '18-29'
        elif entry['age'] < 45:
            entry['age_group'] = '30-44'
        elif entry['age'] < 60:
            entry['age_group'] = '45-59'
        else:
            entry['age_group'] = '60+'
            
        data.append(entry)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_regional_data(100)
    
    # Save to CSV
    output_path = "../data/regional_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records of regional health data and saved to {output_path}")