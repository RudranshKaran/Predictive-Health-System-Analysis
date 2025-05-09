"""
Dynamic Clinical Reference Ranges

This file contains WHO and CDC reference ranges for different blood parameters based on age, gender,
and other demographic factors. These ranges are used to dynamically determine whether a value
is normal, low, or high for a specific patient.
"""

import numpy as np
import pandas as pd

def get_hemoglobin_range(age, gender, pregnant=False):
    """
    Get WHO/CDC reference ranges for hemoglobin based on age, gender and pregnancy status.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
    pregnant : bool, optional
        Whether the female patient is pregnant (default is False)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in g/dL
    """
    # Children reference ranges
    if age < 1:
        return (11.0, 14.0)  # 6-12 months
    elif age < 6:
        return (11.5, 14.0)  # 1-5 years
    elif age < 12:
        return (11.5, 15.5)  # 6-11 years
    
    # Adolescents (12-18)
    elif age < 18:
        if gender == 1:  # Male
            return (13.0, 16.0)
        else:  # Female
            return (12.0, 15.0)
    
    # Adults
    else:
        if gender == 1:  # Male
            if age >= 65:
                return (12.5, 17.0)  # Elderly males
            else:
                return (13.5, 17.5)  # Adult males
        else:  # Female
            if pregnant:
                # Pregnancy-specific ranges by trimester could be added here
                return (11.0, 14.0)  # General pregnancy range
            elif age >= 65:
                return (11.5, 16.0)  # Elderly females
            else:
                return (12.0, 15.5)  # Adult females

def get_mcv_range(age, gender):
    """
    Get reference ranges for Mean Corpuscular Volume (MCV) based on age and gender.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in fL
    """
    # Children reference ranges
    if age < 1:
        return (70.0, 86.0)
    elif age < 6:
        return (73.0, 89.0)
    elif age < 12:
        return (75.0, 91.0)
    
    # Adolescents and adults
    elif age < 18:
        return (78.0, 98.0)
    else:
        if gender == 1 and age >= 60:  # Elderly males
            return (80.0, 100.0)
        elif gender == 0 and age >= 60:  # Elderly females
            return (81.0, 101.0)
        else:  # Adults
            return (80.0, 100.0)

def get_rbc_range(age, gender):
    """
    Get reference ranges for Red Blood Cell (RBC) count based on age and gender.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in millions/μL
    """
    # Children reference ranges
    if age < 1:
        return (3.8, 5.5)
    elif age < 6:
        return (4.0, 5.2)
    elif age < 12:
        return (4.0, 5.4)
    
    # Adolescents and adults
    elif age < 18:
        if gender == 1:  # Male
            return (4.5, 5.3)
        else:  # Female
            return (4.1, 5.1)
    else:
        if gender == 1:  # Male
            return (4.5, 5.9)
        else:  # Female
            return (4.0, 5.2)

def get_rdw_range(age, gender):
    """
    Get reference ranges for Red Cell Distribution Width (RDW) based on age and gender.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in percentage (%)
    """
    # Children and adolescents
    if age < 18:
        return (11.5, 14.5)
    
    # Adults
    else:
        return (11.5, 14.5)  # Same for adult males and females

def get_mch_range(age, gender):
    """
    Get reference ranges for Mean Corpuscular Hemoglobin (MCH) based on age and gender.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in picograms (pg)
    """
    # Children reference ranges
    if age < 1:
        return (23.0, 31.0)
    elif age < 6:
        return (24.0, 30.0)
    elif age < 12:
        return (25.0, 33.0)
    
    # Adolescents and adults
    else:
        return (27.0, 33.0)  # Same for adult males and females

def get_mchc_range(age, gender):
    """
    Get reference ranges for Mean Corpuscular Hemoglobin Concentration (MCHC) based on age and gender.
    
    Parameters:
    -----------
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
        
    Returns:
    --------
    tuple
        (lower_limit, upper_limit) in g/dL
    """
    # Generally consistent across ages and genders
    return (32.0, 36.0)

def evaluate_parameter(value, param_name, age, gender, pregnant=False):
    """
    Evaluate whether a parameter value is within normal range, low, or high.
    
    Parameters:
    -----------
    value : float
        The measured value of the parameter
    param_name : str
        Name of the parameter (e.g., 'Hemoglobin', 'MCV')
    age : float
        Age of the patient in years
    gender : int
        Gender of the patient (0 for female, 1 for male)
    pregnant : bool, optional
        Whether the female patient is pregnant (default is False)
        
    Returns:
    --------
    dict
        A dictionary with 'status' (Normal/Low/High), 'range', and 'value'
    """
    range_functions = {
        'Hemoglobin': lambda a, g, p=False: get_hemoglobin_range(a, g, p),
        'MCV': lambda a, g, _=False: get_mcv_range(a, g),
        'RBC': lambda a, g, _=False: get_rbc_range(a, g),
        'RDW': lambda a, g, _=False: get_rdw_range(a, g),
        'MCH': lambda a, g, _=False: get_mch_range(a, g),
        'MCHC': lambda a, g, _=False: get_mchc_range(a, g)
    }
    
    if param_name not in range_functions:
        return {
            'status': 'Unknown',
            'range': 'No reference range available',
            'value': value
        }
    
    lower_limit, upper_limit = range_functions[param_name](age, gender, pregnant)
    
    if value < lower_limit:
        status = 'Low'
    elif value > upper_limit:
        status = 'High'
    else:
        status = 'Normal'
    
    return {
        'status': status,
        'range': f'{lower_limit:.1f}-{upper_limit:.1f}',
        'value': value,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit
    }

def get_all_parameter_evaluations(patient_data):
    """
    Evaluate all relevant blood parameters for a patient.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient information and blood test results
        
    Returns:
    --------
    dict
        A dictionary with evaluations for each parameter
    """
    age = patient_data.get('Age', 30)  # Default to 30 if age not provided
    gender = patient_data.get('Gender', 1)  # Default to male if gender not provided
    pregnant = patient_data.get('Pregnant', False)
    
    results = {}
    parameters = ['Hemoglobin', 'RBC', 'MCV', 'MCH', 'MCHC', 'RDW']
    
    for param in parameters:
        if param in patient_data:
            results[param] = evaluate_parameter(
                patient_data[param], param, age, gender, pregnant
            )
    
    return results