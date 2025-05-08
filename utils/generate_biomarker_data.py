import pandas as pd
import random
from datetime import datetime, timedelta

def generate_biomarker_data():
    # Read patient data to get patient IDs and visit dates
    patient_data = pd.read_csv("../data/patient_data.csv")
    unique_patients = patient_data['patient_id'].unique()
    
    # Define normal ranges and variation for each biomarker
    biomarker_ranges = {
        'cholesterol': {'min': 150, 'max': 200, 'std': 20},
        'blood_pressure': {'min': 90, 'max': 120, 'std': 15},
        'glucose': {'min': 70, 'max': 100, 'std': 15},
        'hemoglobin': {'min': 12, 'max': 16, 'std': 1},
        'white_blood_cells': {'min': 4500, 'max': 11000, 'std': 1000}
    }
    
    data = []
    
    for patient_id in unique_patients:
        # Get patient's conditions
        patient_conditions = patient_data[patient_data['patient_id'] == patient_id]['diagnosis'].unique()
        
        # Get patient's visit dates
        visit_dates = sorted(patient_data[patient_data['patient_id'] == patient_id]['visit_date'].unique())
        
        # Generate baseline values based on conditions
        baseline = {
            'cholesterol': random.uniform(
                biomarker_ranges['cholesterol']['min'], 
                biomarker_ranges['cholesterol']['max']
            ),
            'blood_pressure': random.uniform(
                biomarker_ranges['blood_pressure']['min'],
                biomarker_ranges['blood_pressure']['max']
            ),
            'glucose': random.uniform(
                biomarker_ranges['glucose']['min'],
                biomarker_ranges['glucose']['max']
            ),
            'hemoglobin': random.uniform(
                biomarker_ranges['hemoglobin']['min'],
                biomarker_ranges['hemoglobin']['max']
            ),
            'white_blood_cells': random.uniform(
                biomarker_ranges['white_blood_cells']['min'],
                biomarker_ranges['white_blood_cells']['max']
            )
        }
        
        # Adjust baseline based on conditions
        if 'Diabetes Type 2' in patient_conditions:
            baseline['glucose'] += random.uniform(20, 50)
        if 'Heart Disease' in patient_conditions:
            baseline['cholesterol'] += random.uniform(20, 40)
        if 'Hypertension' in patient_conditions:
            baseline['blood_pressure'] += random.uniform(20, 40)
        
        # Generate records for each visit
        for visit_date in visit_dates:
            record = {
                'patient_id': patient_id,
                'date': visit_date,
                'cholesterol': round(random.gauss(baseline['cholesterol'], biomarker_ranges['cholesterol']['std']), 1),
                'blood_pressure': round(random.gauss(baseline['blood_pressure'], biomarker_ranges['blood_pressure']['std'])),
                'glucose': round(random.gauss(baseline['glucose'], biomarker_ranges['glucose']['std']), 1),
                'hemoglobin': round(random.gauss(baseline['hemoglobin'], biomarker_ranges['hemoglobin']['std']), 1),
                'white_blood_cells': round(random.gauss(baseline['white_blood_cells'], biomarker_ranges['white_blood_cells']['std']))
            }
            data.append(record)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate biomarker data
    df = generate_biomarker_data()
    
    # Save to CSV
    output_path = "../data/biomarker_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated biomarker data for {len(df)} visits and saved to {output_path}")