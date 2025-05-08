import pandas as pd
import random
from datetime import datetime, timedelta

def generate_patient_data(num_patients=40):
    # Define conditions and their corresponding medications
    conditions = {
        'Hypertension': {
            'medications': ['Lisinopril', 'Amlodipine', 'Losartan'],
            'follow_up_days': 14,  # Regular blood pressure checks
            'related_conditions': ['Heart Disease', 'Diabetes Type 2']
        },
        'Diabetes Type 2': {
            'medications': ['Metformin', 'Glipizide', 'Januvia'],
            'follow_up_days': 30,
            'related_conditions': ['Obesity', 'Heart Disease']
        },
        'Obesity': {
            'medications': ['Orlistat', 'Phentermine', 'Lifestyle Changes'],
            'follow_up_days': 21,
            'related_conditions': ['Diabetes Type 2', 'Hypertension']
        },
        'Respiratory Infection': {
            'medications': ['Azithromycin', 'Amoxicillin', 'Doxycycline'],
            'follow_up_days': 7,  # Short-term condition
            'related_conditions': ['Asthma']
        },
        'Arthritis': {
            'medications': ['Ibuprofen', 'Celebrex', 'Naproxen'],
            'follow_up_days': 30,
            'related_conditions': []
        },
        'Anxiety': {
            'medications': ['Sertraline', 'Alprazolam', 'Buspirone'],
            'follow_up_days': 28,
            'related_conditions': ['Depression']
        },
        'Depression': {
            'medications': ['Fluoxetine', 'Sertraline', 'Bupropion'],
            'follow_up_days': 28,
            'related_conditions': ['Anxiety']
        },
        'Asthma': {
            'medications': ['Albuterol', 'Fluticasone', 'Montelukast'],
            'follow_up_days': 60,
            'related_conditions': ['Respiratory Infection']
        },
        'Heart Disease': {
            'medications': ['Aspirin', 'Metoprolol', 'Atorvastatin'],
            'follow_up_days': 21,
            'related_conditions': ['Hypertension']
        },
        'Allergies': {
            'medications': ['Cetirizine', 'Loratadine', 'Fexofenadine'],
            'follow_up_days': 45,
            'related_conditions': ['Asthma']
        }
    }
    
    # Generate data
    data = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 31)
    date_range = (end_date - start_date).days
    
    # Generate unique patient IDs
    patient_ids = [f'P{random.randint(1000, 9999)}' for _ in range(num_patients)]
    
    for patient_id in patient_ids:
        # Patient demographics
        patient_age = random.randint(18, 85)
        patient_gender = random.choice(['M', 'F'])
        
        # Assign 2-3 chronic conditions to each patient
        primary_conditions = random.sample(list(conditions.keys()), random.randint(2, 3))
        
        # Generate visits throughout the period
        current_date = start_date
        last_visit_dates = {condition: None for condition in primary_conditions}
        patient_medications = {condition: None for condition in primary_conditions}
        
        while current_date <= end_date:
            for condition in primary_conditions:
                # Check if it's time for a follow-up
                if (last_visit_dates[condition] is None or 
                    (current_date - last_visit_dates[condition]).days >= conditions[condition]['follow_up_days']):
                    
                    # Randomly adjust visit date by ±3 days to make it more realistic
                    visit_date = current_date + timedelta(days=random.randint(-3, 3))
                    if start_date <= visit_date <= end_date:  # Ensure date is within range
                        # Sometimes switch medications for better management
                        if patient_medications[condition]:
                            if random.random() < 0.3:  # 30% chance to switch medication
                                new_meds = [m for m in conditions[condition]['medications'] 
                                          if m != patient_medications[condition]]
                                medication = random.choice(new_meds if new_meds else conditions[condition]['medications'])
                            else:
                                medication = patient_medications[condition]  # Continue same medication
                        else:
                            medication = random.choice(conditions[condition]['medications'])
                            
                        patient_medications[condition] = medication
                        
                        data.append({
                            'patient_id': patient_id,
                            'visit_date': visit_date.strftime('%Y-%m-%d'),
                            'diagnosis': condition,
                            'medications': medication,
                            'age': patient_age,
                            'gender': patient_gender
                        })
                        
                        last_visit_dates[condition] = visit_date
                        
                        # Chance of related condition appearing
                        if conditions[condition]['related_conditions']:
                            if random.random() < 0.15:  # 15% chance
                                related_condition = random.choice(conditions[condition]['related_conditions'])
                                if related_condition not in primary_conditions:
                                    data.append({
                                        'patient_id': patient_id,
                                        'visit_date': (visit_date + timedelta(days=random.randint(1, 5))).strftime('%Y-%m-%d'),
                                        'diagnosis': related_condition,
                                        'medications': random.choice(conditions[related_condition]['medications']),
                                        'age': patient_age,
                                        'gender': patient_gender
                                    })
            
            # Move to next week
            current_date += timedelta(days=7)
    
    # Convert to DataFrame and sort by date
    df = pd.DataFrame(data)
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_patient_data(40)  # Generate data for 40 patients
    
    # Save to CSV
    output_path = "../data/patient_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} patient visit records and saved to {output_path}")