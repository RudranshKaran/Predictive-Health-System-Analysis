import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # Add SHAP library import
import warnings
import os
warnings.filterwarnings('ignore')

# Get the absolute path to the model directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model')

# Load the saved model and scaler with absolute paths
model_path = os.path.join(model_dir, 'early_anemia_model.pkl')
scaler_path = os.path.join(model_dir, 'early_anemia_scaler.pkl')

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Create a SHAP explainer for the model - with improved error handling
if hasattr(model, 'predict_proba'):
    # For tree-based models (RandomForest, GradientBoosting)
    try:
        # First attempt tree explainer which is faster and more accurate for tree-based models
        explainer = shap.TreeExplainer(model)
        print("Using TreeExplainer for SHAP analysis")
    except Exception as e:
        print(f"TreeExplainer failed: {str(e)}")
        try:
            # Fallback to Permutation explainer which is more robust for different model types
            # Create a small background dataset for the explainer
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            n_features = len(feature_names) if feature_names is not None else 12
            background_data = np.zeros((10, n_features))
            
            # Define a predict function that returns probabilities for the positive class only
            def predict_proba_positive_class(X):
                return model.predict_proba(X)[:, 1]
            
            explainer = shap.PermutationExplainer(
                predict_proba_positive_class,
                background_data,
                feature_names=feature_names
            )
            print("Using PermutationExplainer as fallback for SHAP analysis")
        except Exception as e:
            print(f"All SHAP explainers failed: {str(e)}")
            explainer = None
else:
    print("Model does not support SHAP analysis")
    explainer = None

def anemia_type_detector(gender, hb, mcv, mch, mchc, rdw=None, rbc=None, age=None):
    """
    Determine the most likely anemia type based on blood parameters
    
    Parameters:
    - gender: 1 for male, 0 for female
    - hb: Hemoglobin level
    - mcv: Mean Corpuscular Volume
    - mch: Mean Corpuscular Hemoglobin
    - mchc: Mean Corpuscular Hemoglobin Concentration
    - rdw: Red Cell Distribution Width (if available)
    - rbc: Red Blood Cell count (if available)
    - age: Patient's age in years (if available)
    
    Returns:
    - anemia_type: string name of the likely anemia type
    - confidence: float between 0-1 indicating confidence in the type assessment
    - description: string description of the identified anemia
    """
    # Define gender-specific Hb thresholds
    hb_low = 13.0 if gender == 1 else 12.0
    
    # Define normal RDW range
    rdw_normal_range = (11.5, 14.5)
    
    # Define normal RBC range (gender-specific)
    rbc_normal_range = (4.5, 5.9) if gender == 1 else (4.0, 5.2)
    
    # Age-specific adjustments (older adults may have lower baseline hemoglobin)
    if age is not None and age > 65:
        hb_low -= 0.3  # Slightly lower threshold for elderly
    
    # Check for Iron Deficiency Anemia (microcytic, hypochromic)
    if mcv < 80 and mch < 27 and mchc < 32:
        confidence = 0.8
        description = 'Low MCV, MCH, MCHC'
        
        # If RDW is available, use it to differentiate iron deficiency from thalassemia
        if rdw is not None:
            if rdw > rdw_normal_range[1]:  # Elevated RDW suggests iron deficiency
                confidence += 0.1
                description += f" with elevated RDW ({rdw:.1f}%) strongly suggests iron deficiency"
            else:  # Normal RDW with microcytic pattern suggests thalassemia
                confidence -= 0.1
                description += f" with normal RDW ({rdw:.1f}%) suggests possible thalassemia trait"
        
        # If RBC is available, use it for further discrimination
        if rbc is not None:
            if rbc < rbc_normal_range[0]:  # Low RBC count supports iron deficiency
                confidence += 0.05
                description += f", low RBC count ({rbc:.2f}) supports iron deficiency"
            elif rbc >= rbc_normal_range[0]:  # Normal/High RBC count suggests thalassemia
                confidence -= 0.1
                description += f", normal/high RBC count ({rbc:.2f}) suggests possible thalassemia trait"
                return {
                    'type': 'Possible Thalassemia Trait', 
                    'confidence': confidence,
                    'description': description
                }
                
        return {
            'type': 'Iron Deficiency Anemia', 
            'confidence': confidence,
            'description': description
        }
        
    # Check for Vitamin B12 / Folate Deficiency (macrocytic)
    elif mcv > 100:
        # High RDW strengthens this diagnosis
        confidence = 0.75 if mcv > 105 else 0.6
        description = 'Elevated MCV suggests vitamin B12 or folate deficiency'
        
        if rdw is not None and rdw > rdw_normal_range[1]:
            confidence += 0.1
            description += f", supported by elevated RDW ({rdw:.1f}%)"
        
        # Age factor - B12 deficiency more common in elderly
        if age is not None and age > 60:
            confidence += 0.05
            description += ", more common in older adults"
            
        return {
            'type': 'B12/Folate Deficiency', 
            'confidence': confidence,
            'description': description
        }
        
    # Check for Thalassemia Trait (microcytic but often with normal MCHC and normal RDW)
    elif mcv < 80 and mch < 27 and mchc >= 32:
        confidence = 0.55
        description = 'Microcytic with relatively normal MCHC'
        
        if rdw is not None and rdw <= rdw_normal_range[1]:  # Normal RDW is characteristic of thalassemia trait
            confidence += 0.2
            description += f', normal RDW ({rdw:.1f}%) strongly suggests thalassemia trait'
        
        # RBC count is often elevated in thalassemia trait
        if rbc is not None and rbc > rbc_normal_range[1]:
            confidence += 0.15
            description += f', elevated RBC count ({rbc:.2f}) very characteristic of thalassemia'
            
        return {
            'type': 'Possible Thalassemia Trait', 
            'confidence': confidence,
            'description': description
        }
        
    # Check for Anemia of Chronic Disease (often normocytic, normochromic)
    elif hb < hb_low and 80 <= mcv <= 100 and mch >= 26:
        confidence = 0.5
        description = 'Normal MCV with low Hb suggests anemia of chronic disease'
        
        if rdw is not None and rdw <= rdw_normal_range[1]:  # Normal RDW in ACD
            confidence += 0.1
            description += f', supported by normal RDW ({rdw:.1f}%)'
        
        # Age factor - ACD more common in older adults
        if age is not None and age > 50:
            confidence += 0.1
            description += ', more common in older adults'
            
        return {
            'type': 'Anemia of Chronic Disease', 
            'confidence': confidence,
            'description': description
        }
        
    # Check for Hemolytic Anemia (often normocytic with elevated RDW)
    elif hb < hb_low and 80 <= mcv <= 100 and mchc > 35:
        confidence = 0.4
        description = 'Normal MCV with elevated MCHC'
        
        if rdw is not None and rdw > rdw_normal_range[1]:
            confidence += 0.2
            description += f' and elevated RDW ({rdw:.1f}%) suggests hemolysis'
            
        return {
            'type': 'Possible Hemolytic Anemia', 
            'confidence': confidence,
            'description': description
        }
        
    # Early-stage / Latent Anemia types
    elif hb >= hb_low:
        if mcv < 82 and mch < 28:
            confidence = 0.6
            description = 'Normal Hb with low MCV/MCH'
            
            if rdw is not None and rdw > rdw_normal_range[1]:  # Early iron deficiency often shows RDW elevation first
                confidence += 0.15
                description += f' and elevated RDW ({rdw:.1f}%) strongly suggests early iron deficiency'
            
            # RBC count can help in early detection
            if rbc is not None and rbc < rbc_normal_range[0] + 0.3:
                confidence += 0.1
                description += f', borderline low RBC count ({rbc:.2f}) supports early deficiency'
                
            return {
                'type': 'Latent Iron Deficiency', 
                'confidence': confidence,
                'description': description
            }
        elif mcv > 98:
            confidence = 0.45
            description = 'Normal Hb with high-normal MCV may indicate early B12/folate deficiency'
            
            # Age-related adjustment for B12 deficiency risk
            if age is not None and age > 60:
                confidence += 0.1
                description += ', particularly concerning in older adults'
                
            return {
                'type': 'Latent B12/Folate Deficiency', 
                'confidence': confidence,
                'description': description
            }
        
    # Default case
    return {
        'type': 'Unspecified or Mixed Pattern', 
        'confidence': 0.3,
        'description': 'Blood parameters do not clearly indicate a specific anemia type'
    }

def predict_early_anemia_risk(patient_data, model, scaler, detailed=False):
    """
    Predict early anemia risk for a patient with comprehensive analysis
    
    Parameters:
    patient_data: dict with keys 'Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 
                 and optional 'RDW', 'RBC', 'Age'
    model: trained model
    scaler: fitted scaler
    detailed: whether to return detailed analysis
    
    Returns:
    risk_prediction: binary prediction (0=low risk, 1=high risk)
    risk_probability: probability of developing anemia
    risk_category: risk category (Very Low, Low, Moderate, High)
    analysis: detailed parameter analysis (if detailed=True)
    """
    # Create a copy of patient data for model input
    model_input_data = {k: v for k, v in patient_data.items()}
    
    # Create a DataFrame for the patient
    patient_df = pd.DataFrame([model_input_data])
    
    # Extract feature_names_in_ from scaler if available
    if hasattr(scaler, 'feature_names_in_'):
        training_features = list(scaler.feature_names_in_)
    else:
        # Fallback to our list of expected features in case scaler doesn't have feature_names_in_
        training_features = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender',
                            'RBC_MCV_Ratio', 'RDW_MCV_Ratio', 'MCH_MCHC_Ratio',
                            'Age_RBC_Interaction', 'RDW_Age_Interaction']
    
    # Set default values for missing required base features
    required_features = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender']
    for feature in required_features:
        if feature not in patient_df.columns:
            # Use sensible defaults for missing features
            if feature == 'Age':
                patient_df[feature] = 45  # Middle-aged default
            elif feature == 'RBC':
                # Gender-specific default RBC values
                if patient_data.get('Gender') == 1:  # Male
                    patient_df[feature] = 5.0  # Normal male RBC
                else:  # Female
                    patient_df[feature] = 4.5  # Normal female RBC
            elif feature == 'RDW':
                patient_df[feature] = 13.0  # Normal RDW
            else:
                patient_df[feature] = 0  # Default for other features
    
    # Remove Hemoglobin and other non-training features
    columns_to_drop = ['Hemoglobin', 'MCH_MCV_Ratio', 'MCHC_MCV_Ratio']
    for col in columns_to_drop:
        if col in patient_df.columns:
            patient_df.drop(columns=[col], inplace=True, errors='ignore')
    
    # Add the exact derived features used during training
    if 'RBC' in patient_df.columns and 'MCV' in patient_df.columns:
        patient_df['RBC_MCV_Ratio'] = patient_df['RBC'] / patient_df['MCV']
    if 'RDW' in patient_df.columns and 'MCV' in patient_df.columns:
        patient_df['RDW_MCV_Ratio'] = patient_df['RDW'] / patient_df['MCV']
    if 'MCH' in patient_df.columns and 'MCHC' in patient_df.columns:
        patient_df['MCH_MCHC_Ratio'] = patient_df['MCH'] / patient_df['MCHC']
    if 'Age' in patient_df.columns and 'RBC' in patient_df.columns:
        patient_df['Age_RBC_Interaction'] = patient_df['Age'] * patient_df['RBC'] / 100
    if 'Age' in patient_df.columns and 'RDW' in patient_df.columns:
        patient_df['RDW_Age_Interaction'] = patient_df['RDW'] * np.log(patient_df['Age'] + 1)
    
    # Ensure the DataFrame has exactly the same features AND ORDER as used in training
    patient_df = patient_df.reindex(columns=training_features, fill_value=0)
    
    # Scale the features using the same scaler used during training
    patient_scaled = pd.DataFrame(
        scaler.transform(patient_df),
        columns=patient_df.columns
    )
    
    # Make predictions with the ML model
    ml_prediction = model.predict(patient_scaled)[0]
    ml_probability = model.predict_proba(patient_scaled)[0, 1]
    
    # Prepare data for pattern detection and rule-based adjustments
    gender = patient_data['Gender']
    hb = patient_data['Hemoglobin']
    mcv = patient_data['MCV']
    mch = patient_data['MCH']
    mchc = patient_data['MCHC']
    rdw = patient_data.get('RDW')  # Get RDW if available, else None
    rbc = patient_data.get('RBC')  # Get RBC if available, else None
    age = patient_data.get('Age')  # Get Age if available, else None
    
    # Reference ranges (approximate normal values)
    reference_ranges = {
        'Hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 15.5)},
        'MCH': (27.0, 33.0),
        'MCHC': (32.0, 36.0),
        'MCV': (80.0, 100.0),
        'RDW': (11.5, 14.5),
        'RBC': {'male': (4.5, 5.9), 'female': (4.0, 5.2)}
    }
    gender_str = 'male' if gender == 1 else 'female'
    hb_range = reference_ranges['Hemoglobin'][gender_str]
    
    # Detect blood parameter patterns
    patterns = []
    
    # Microcytic hypochromic pattern (iron deficiency anemia)
    if mcv < reference_ranges['MCV'][0] and (mch < reference_ranges['MCH'][0] or mchc < reference_ranges['MCHC'][0]):
        # Initial risk adjustment
        risk_adj = 0.4
        description = "Suggestive of iron deficiency anemia or thalassemia"
        
        # Use RDW to differentiate between iron deficiency and thalassemia
        if rdw is not None:
            if rdw > reference_ranges['RDW'][1]:
                risk_adj = 0.5  # Higher risk adjustment for elevated RDW (suggests iron deficiency)
                description += f" (elevated RDW of {rdw:.1f}% strongly suggests iron deficiency)"
            else:
                description += f" (normal RDW of {rdw:.1f}% suggests possible thalassemia trait)"
        
        # Use RBC to further differentiate (normal/high in thalassemia, low in iron deficiency)
        if rbc is not None:
            rbc_range = reference_ranges['RBC'][gender_str]
            if rbc < rbc_range[0]:  # Low RBC
                risk_adj += 0.1  # Increase risk (supports iron deficiency)
                description += f", low RBC count ({rbc:.2f}) supports iron deficiency"
            elif rbc > rbc_range[1]:  # High RBC
                risk_adj -= 0.1  # Decrease risk (suggests thalassemia)
                description += f", elevated RBC count ({rbc:.2f}) suggests thalassemia trait"
                
        patterns.append({
            'name': "MICROCYTIC HYPOCHROMIC PATTERN", 
            'description': description,
            'risk_adjustment': risk_adj
        })
    
    # Macrocytic pattern (B12/folate deficiency)
    if mcv > reference_ranges['MCV'][1]:
        risk_adj = 0.45
        description = "Suggestive of B12/folate deficiency"
        
        if rdw is not None and rdw > reference_ranges['RDW'][1]:
            risk_adj = 0.55  # Higher risk with elevated RDW
            description += f" (supported by elevated RDW of {rdw:.1f}%)"
        
        # Age factor - B12 deficiency more common in elderly
        if age is not None and age > 60:
            risk_adj += 0.1
            description += f", particularly concerning at age {age}"
            
        patterns.append({
            'name': "MACROCYTIC PATTERN", 
            'description': description,
            'risk_adjustment': risk_adj
        })
    
    # Normocytic normochromic pattern (anemia of chronic disease)
    if (reference_ranges['MCV'][0] <= mcv <= reference_ranges['MCV'][1]) and \
       (reference_ranges['MCH'][0] <= mch <= reference_ranges['MCH'][1]) and \
       hb < hb_range[0]:
        risk_adj = 0.35
        description = "May indicate anemia of chronic disease"
        
        # Age factor - ACD more common in older adults
        if age is not None and age > 50:
            risk_adj += 0.1
            description += f", more common in older adults (patient age: {age})"
            
        patterns.append({
            'name': "NORMOCYTIC NORMOCHROMIC PATTERN", 
            'description': description,
            'risk_adjustment': risk_adj
        })
    
    # Early-stage latent anemia pattern (normal Hb, abnormal indices)
    if hb >= hb_range[0] and (
        (mcv < reference_ranges['MCV'][0] + 5 and mch < reference_ranges['MCH'][0] + 3) or  # Early microcytic
        (mchc < reference_ranges['MCHC'][0] + 2) or  # Early hypochromic
        (hb/mch > 0.6)  # Imbalanced Hb/MCH ratio
    ):
        risk_adj = 0.3
        description = "Normal hemoglobin but RBC indices suggest developing anemia"
        
        # RDW elevation can be an early sign of developing anemia
        if rdw is not None and rdw > reference_ranges['RDW'][1]:
            risk_adj = 0.4
            description += f" (elevated RDW of {rdw:.1f}% suggests active erythropoietic changes)"
        
        # RBC count can help in early detection
        if rbc is not None:
            rbc_range = reference_ranges['RBC'][gender_str]
            if rbc < rbc_range[0] + 0.2:  # Slightly low or borderline low RBC
                risk_adj += 0.1
                description += f", borderline low RBC count ({rbc:.2f}) supports early deficiency"
            
        patterns.append({
            'name': "LATENT ANEMIA PATTERN", 
            'description': description,
            'risk_adjustment': risk_adj
        })
    
    # Age-specific risk factor (elderly have higher risk of anemia)
    if age is not None and age > 65:
        risk_adj = 0.15
        description = f"Advanced age ({age} years) increases baseline anemia risk"
        
        # Higher risk with borderline parameters
        if hb < hb_range[1] - 2:  # Within 2 g/dL of lower normal limit
            risk_adj = 0.25
            description += f", especially with borderline hemoglobin ({hb:.1f} g/dL)"
            
        patterns.append({
            'name': "AGE-RELATED RISK FACTOR", 
            'description': description,
            'risk_adjustment': risk_adj
        })
    
    # Determine anemia type based on the values
    anemia_assessment = anemia_type_detector(gender, hb, mcv, mch, mchc, rdw, rbc, age)
    
    # HYBRID APPROACH: Adjust ML probability based on clinical patterns and anemia type
    adjusted_probability = ml_probability
    
    # Set minimum risk floor based on detected patterns
    for pattern in patterns:
        adjusted_probability = max(adjusted_probability, pattern['risk_adjustment'])
    
    # Further adjust probability based on anemia type assessment
    if anemia_assessment['confidence'] > 0.5:
        if "Iron Deficiency" in anemia_assessment['type'] or "Thalassemia" in anemia_assessment['type']:
            adjusted_probability = max(adjusted_probability, anemia_assessment['confidence'] * 0.7)
        elif "B12/Folate" in anemia_assessment['type']:
            adjusted_probability = max(adjusted_probability, anemia_assessment['confidence'] * 0.6)
        elif "Latent" in anemia_assessment['type']:
            adjusted_probability = max(adjusted_probability, anemia_assessment['confidence'] * 0.5)
    
    # Determine adjusted prediction and risk category
    adjusted_prediction = 1 if adjusted_probability >= 0.3 else 0
    
    # Determine risk category based on adjusted probability
    if adjusted_probability < 0.25:
        risk_category = "Very Low Risk"
    elif adjusted_probability < 0.5:
        risk_category = "Low Risk"
    elif adjusted_probability < 0.75:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
    
    if not detailed:
        return adjusted_prediction, adjusted_probability, risk_category
    
    # Prepare detailed analysis
    analysis = {}
    
    # Individual parameter analysis
    analysis['Hemoglobin'] = {
        'value': hb,
        'status': "LOW - contributes strongly to anemia risk" if hb < hb_range[0] else
                 "BORDERLINE LOW - contributes to anemia risk" if hb <= hb_range[0] + 0.5 else
                 "HIGH - contributes negatively to anemia risk" if hb > hb_range[1] else
                 "NORMAL - contributes negatively to anemia risk",
        'range': hb_range
    }
    
    mch_range = reference_ranges['MCH']
    analysis['MCH'] = {
        'value': mch,
        'status': "LOW - suggests hypochromic anemia pattern" if mch < mch_range[0] else
                 "HIGH - suggests macrocytic pattern" if mch > mch_range[1] else
                 "NORMAL",
        'range': mch_range
    }
    
    mchc_range = reference_ranges['MCHC']
    analysis['MCHC'] = {
        'value': mchc,
        'status': "LOW - indicates hypochromic RBCs" if mchc < mchc_range[0] else
                 "HIGH - unusual, may indicate lab error" if mchc > mchc_range[1] else
                 "NORMAL",
        'range': mchc_range
    }
    
    mcv_range = reference_ranges['MCV']
    analysis['MCV'] = {
        'value': mcv,
        'status': "LOW - suggests microcytic pattern" if mcv < mcv_range[0] else
                 "HIGH - suggests macrocytic pattern" if mcv > mcv_range[1] else
                 "NORMAL",
        'range': mcv_range
    }
    
    # Add RDW analysis if available
    if rdw is not None:
        rdw_range = reference_ranges['RDW']
        analysis['RDW'] = {
            'value': rdw,
            'status': "ELEVATED - suggests active erythropoiesis/heterogeneous RBC population" if rdw > rdw_range[1] else
                     "LOW - unusual finding" if rdw < rdw_range[0] else
                     "NORMAL - homogeneous RBC population",
            'range': rdw_range
        }
    
    # Add RBC analysis if available
    if rbc is not None:
        rbc_range = reference_ranges['RBC'][gender_str]
        analysis['RBC'] = {
            'value': rbc,
            'status': "LOW - suggests decreased RBC production or increased destruction" if rbc < rbc_range[0] else
                     "HIGH - suggests possible polycythemia or dehydration" if rbc > rbc_range[1] else
                     "NORMAL",
            'range': rbc_range
        }
    
    # Add Age analysis if available
    if age is not None:
        analysis['Age'] = {
            'value': age,
            'status': "ELDERLY - increases baseline anemia risk" if age > 65 else
                     "MIDDLE-AGED - moderate baseline risk" if age > 40 else
                     "YOUNG - lower baseline risk"
        }
    
    # Include detected patterns
    analysis['patterns'] = patterns
    
    # Include anemia type assessment
    analysis['anemia_type'] = anemia_assessment
    
    # Include both ML and adjusted probabilities for comparison
    analysis['ml_probability'] = ml_probability
    analysis['adjusted_probability'] = adjusted_probability
    
    return adjusted_prediction, adjusted_probability, risk_category, analysis

def print_detailed_analysis(patient_data, prediction=None, probability=None, category=None, analysis=None):
    """Print a detailed clinical analysis of the blood parameters"""
    if analysis is None:
        _, _, _, analysis = predict_early_anemia_risk(patient_data, model, scaler, detailed=True)
    
    gender = "Male" if patient_data['Gender'] == 1 else "Female"
    gender_str = "male" if patient_data['Gender'] == 1 else "female"
    
    print("\n==================================================")
    print("           COMPREHENSIVE BLOOD ANALYSIS           ")
    print("==================================================")
    print(f"Patient Gender: {gender}")
    print("\n----- INDIVIDUAL PARAMETER ANALYSIS -----")
    
    # Hemoglobin
    hb_info = analysis['Hemoglobin']
    print(f"\nHemoglobin: {hb_info['value']:.1f} g/dL ({hb_info['range'][0]}-{hb_info['range'][1]})")
    print(f"  Status: {hb_info['status']}")
    
    # MCH
    mch_info = analysis['MCH']
    print(f"\nMCH: {mch_info['value']:.1f} pg ({mch_info['range'][0]}-{mch_info['range'][1]})")
    print(f"  Status: {mch_info['status']}")
    
    # MCHC
    mchc_info = analysis['MCHC']
    print(f"\nMCHC: {mchc_info['value']:.1f} g/dL ({mchc_info['range'][0]}-{mchc_info['range'][1]})")
    print(f"  Status: {mchc_info['status']}")
    
    # MCV
    mcv_info = analysis['MCV']
    print(f"\nMCV: {mcv_info['value']:.1f} fL ({mcv_info['range'][0]}-{mcv_info['range'][1]})")
    print(f"  Status: {mcv_info['status']}")
    
    # RDW (if available)
    if 'RDW' in analysis:
        rdw_info = analysis['RDW']
        print(f"\nRDW: {rdw_info['value']:.1f}% ({rdw_info['range'][0]}-{rdw_info['range'][1]})")
        print(f"  Status: {rdw_info['status']}")
    
    # Derived Ratios
    print("\n----- DERIVED RATIOS -----")
    print(f"Hb/MCH Ratio: {patient_data['Hemoglobin']/patient_data['MCH']:.2f}")
    print(f"Hb/MCHC Ratio: {patient_data['Hemoglobin']/patient_data['MCHC']:.2f}")
    print(f"MCH/MCV Ratio: {patient_data['MCH']/patient_data['MCV']:.3f}")
    
    # Anemia Type Assessment
    anemia_type = analysis['anemia_type']
    print("\n----- ANEMIA TYPE ASSESSMENT -----")
    print(f"Type: {anemia_type['type']}")
    print(f"Confidence: {anemia_type['confidence']:.2f}")
    print(f"Description: {anemia_type['description']}")
    
    # Pattern Analysis
    print("\n----- PATTERN ANALYSIS -----")
    if analysis['patterns']:
        for pattern in analysis['patterns']:
            print(f"• {pattern['name']} - {pattern['description']}")
            print(f"  Risk floor: {pattern['risk_adjustment']}")
    else:
        print("• No specific anemia patterns detected")
    
    # SHAP Analysis - FIXED IMPLEMENTATION
    if explainer is not None and prediction is not None:
        print("\n----- EXPLAINABLE AI ANALYSIS (SHAP) -----")
        
        # Create preprocessing for the patient data to match model input format
        patient_df = pd.DataFrame([patient_data])
        
        # Extract feature_names_in_ from scaler if available
        if hasattr(scaler, 'feature_names_in_'):
            training_features = list(scaler.feature_names_in_)
        else:
            # Fallback to expected features
            training_features = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender',
                              'RBC_MCV_Ratio', 'RDW_MCV_Ratio', 'MCH_MCHC_Ratio',
                              'Age_RBC_Interaction', 'RDW_Age_Interaction']
        
        # Set default values for missing required base features
        required_features = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender']
        for feature in required_features:
            if feature not in patient_df.columns:
                if feature == 'Age':
                    patient_df[feature] = 45
                elif feature == 'RBC':
                    patient_df[feature] = 5.0 if patient_data.get('Gender') == 1 else 4.5
                elif feature == 'RDW':
                    patient_df[feature] = 13.0
                else:
                    patient_df[feature] = 0
        
        # Add derived features
        if 'RBC' in patient_df.columns and 'MCV' in patient_df.columns:
            patient_df['RBC_MCV_Ratio'] = patient_df['RBC'] / patient_df['MCV']
        if 'RDW' in patient_df.columns and 'MCV' in patient_df.columns:
            patient_df['RDW_MCV_Ratio'] = patient_df['RDW'] / patient_df['MCV']
        if 'MCH' in patient_df.columns and 'MCHC' in patient_df.columns:
            patient_df['MCH_MCHC_Ratio'] = patient_df['MCH'] / patient_df['MCHC']
        if 'Age' in patient_df.columns and 'RBC' in patient_df.columns:
            patient_df['Age_RBC_Interaction'] = patient_df['Age'] * patient_df['RBC'] / 100
        if 'Age' in patient_df.columns and 'RDW' in patient_df.columns:
            patient_df['RDW_Age_Interaction'] = patient_df['RDW'] * np.log(patient_df['Age'] + 1)
        
        # Remove Hemoglobin and other non-training features
        columns_to_drop = ['Hemoglobin', 'MCH_MCV_Ratio', 'MCHC_MCV_Ratio']
        for col in columns_to_drop:
            if col in patient_df.columns:
                patient_df.drop(columns=[col], inplace=True, errors='ignore')
        
        # Reindex to ensure correct feature order
        patient_df = patient_df.reindex(columns=training_features, fill_value=0)
        
        # Scale features
        patient_scaled = pd.DataFrame(
            scaler.transform(patient_df),
            columns=patient_df.columns
        )
        
        try:
            # Calculate SHAP values for this patient - IMPROVED ERROR HANDLING
            if isinstance(explainer, shap.TreeExplainer):
                # For TreeExplainer
                shap_values = explainer.shap_values(patient_scaled)
                
                # TreeExplainer might return a list where index 1 is positive class values
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    patient_shap_values = shap_values[1][0]
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value
                else:
                    patient_shap_values = shap_values[0]
                    expected_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]
            
            elif isinstance(explainer, shap.PermutationExplainer):
                # For PermutationExplainer
                shap_values = explainer(patient_scaled)
                patient_shap_values = shap_values.values[0]
                # PermutationExplainer doesn't have expected_value attribute, set to 0.5 default
                expected_value = 0.5  # Default value for binary classification
            
            else:
                # Generic approach
                shap_values = explainer(patient_scaled)
                
                # Check if the result is a shap.Explanation object
                if isinstance(shap_values, shap.Explanation):
                    patient_shap_values = shap_values.values[0]
                    expected_value = shap_values.base_values[0]
                else:
                    # Try to extract values based on shape
                    if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                        patient_shap_values = shap_values[0]
                    else:
                        patient_shap_values = shap_values
                    
                    # Try to get expected value
                    expected_value = getattr(explainer, 'expected_value', 0.5)
            
            # Ensure expected_value is a scalar
            if hasattr(expected_value, '__len__') and not isinstance(expected_value, str):
                expected_value = expected_value[0]
            
            # Get features sorted by absolute SHAP value (most important first)
            feature_importance = sorted(zip(training_features, patient_shap_values), 
                                      key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nBase risk score: {float(expected_value):.4f}")
            print("Top features influencing this patient's risk prediction:")
            
            # Define reference ranges for comparison
            reference_ranges = {
                'Hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 15.5)},
                'MCH': (27.0, 33.0),
                'MCHC': (32.0, 36.0),
                'MCV': (80.0, 100.0),
                'RDW': (11.5, 14.5),
                'RBC': {'male': (4.5, 5.9), 'female': (4.0, 5.2)}
            }
            
            # Show top 5 most important features
            for feature, value in feature_importance[:5]:
                impact = "INCREASES" if value > 0 else "DECREASES"
                if feature in patient_data:
                    original_value = patient_data[feature]
                    # Check if this feature is in our reference ranges
                    if feature in ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW']:
                        if feature == 'RBC':
                            ref_range = reference_ranges['RBC'][gender_str]
                        else:
                            ref_range = reference_ranges[feature]
                        
                        # Determine if value is low, normal or high
                        value_status = "LOW" if original_value < ref_range[0] else \
                                    "HIGH" if original_value > ref_range[1] else \
                                    "normal"
                        
                        print(f"• {feature} = {original_value:.2f} ({value_status}): {impact} risk by {abs(value):.4f}")
                    else:
                        print(f"• {feature} = {original_value:.2f}: {impact} risk by {abs(value):.4f}")
                else:
                    # For derived features
                    print(f"• {feature}: {impact} risk by {abs(value):.4f}")
            
            # Create and save individual patient SHAP plot
            try:
                plt.figure(figsize=(10, 6))
                
                # Create an explanation object for the waterfall plot
                explanation = shap.Explanation(
                    values=patient_shap_values,
                    base_values=float(expected_value),
                    data=patient_scaled.iloc[0].values,
                    feature_names=training_features
                )
                
                shap.plots.waterfall(explanation, show=False)
                plt.title('SHAP Analysis: Why This Patient Received This Risk Score')
                plt.tight_layout()
                plt.savefig('model/patient_analysis_shap_importance.png')
                plt.close()
                
                print("\nSHAP analysis visualization saved to 'model/patient_analysis_shap_importance.png'")
                print("This visualization shows exactly how each feature contributed to the final risk score.")
            except Exception as plot_err:
                print(f"\nCould not generate SHAP plot: {str(plot_err)}")
                print("However, the numerical SHAP values were successfully computed.")
        
        except Exception as e:
            print(f"\nCould not generate SHAP analysis: {str(e)}")
            print("This may happen if the model structure is not compatible with SHAP explainer.")
    
    # Model Prediction
    if prediction is not None:
        print("\n----- MODEL PREDICTION -----")
        print(f"ML Model Risk: {analysis['ml_probability']:.2f}")
        print(f"Adjusted Risk: {probability:.2f} ({category})")
        
        print("\n----- CONCLUSION -----")
        
        if analysis['ml_probability'] < 0.3 and probability >= 0.3:
            print("⚠️  PATTERN-BASED RISK ADJUSTMENT")
            print(f"     ML model predicted low risk ({analysis['ml_probability']:.2f})")
            print(f"     Adjusted to {probability:.2f} based on blood parameter patterns")
            print(f"     Suspected anemia type: {anemia_type['type']}")
            
            if "Latent" in anemia_type['type'] or probability < 0.5:
                print("\n     RECOMMENDATION: Follow-up in 3 months")
                print("     Consider testing ferritin, iron, TIBC, vitamin B12, folate")
            else:
                print("\n     RECOMMENDATION: Follow-up within 1 month")
                print("     Additional testing recommended based on suspected type")
                
        elif probability >= 0.5:
            print("🚨 HIGH ANEMIA RISK DETECTED")
            print(f"     Risk probability: {probability:.2f}")
            print(f"     Suspected type: {anemia_type['type']}")
            print("\n     RECOMMENDATION: Prompt clinical assessment")
            
            if "Iron Deficiency" in anemia_type['type']:
                print("     Consider iron studies, ferritin, TIBC")
            elif "B12/Folate" in anemia_type['type']:
                print("     Consider B12, folate, methylmalonic acid testing")
            elif "Thalassemia" in anemia_type['type']:
                print("     Consider hemoglobin electrophoresis, genetic testing")
                
        else:
            print("✅ LOW RISK PROFILE")
            print(f"     Risk probability: {probability:.2f}")
            print("     Routine follow-up recommended")
    
    # Feature importance insights
    print("\n----- FEATURE IMPORTANCE INSIGHTS -----")
    # Add generic insights based on known medical relationships
    if 'MCV' in patient_data and patient_data['MCV'] < 80:
        print(f"• Low MCV ({patient_data['MCV']:.1f}) indicates microcytic pattern")
    elif 'MCV' in patient_data and patient_data['MCV'] > 100:
        print(f"• High MCV ({patient_data['MCV']:.1f}) indicates macrocytic pattern")
        
    if 'RDW' in patient_data and patient_data['RDW'] > 15:
        print(f"• Elevated RDW ({patient_data['RDW']:.1f}%) suggests active erythropoiesis")
        
    if 'MCH' in patient_data and patient_data['MCH'] < 27:
        print(f"• Low MCH ({patient_data['MCH']:.1f}) contributes to increased risk")
    
    print("\n==================================================")

# Add test cases specifically for different anemia types
test_cases = {
    "iron_deficiency": {
        'Gender': 1,  # Male
        'Hemoglobin': 12.0,  # Low (anemic)
        'MCH': 22.0,  # Low (hypochromic)
        'MCHC': 29.0,  # Low (hypochromic)
        'MCV': 75.0,   # Low (microcytic)
        'RDW': 16.5,   # Elevated (typical in iron deficiency)
        'RBC': 4.1,    # Slightly low (typical in iron deficiency)
        'Age': 45      # Middle-aged
    },
    "latent_iron_deficiency": {
        'Gender': 1,  # Male
        'Hemoglobin': 16.0,  # Normal/high
        'MCH': 22.0,  # Low (hypochromic)
        'MCHC': 29.0,  # Low (hypochromic)
        'MCV': 82.0,   # Low-normal
        'RDW': 15.2,   # Slightly elevated (early changes)
        'RBC': 4.7,    # Normal
        'Age': 35      # Young adult
    },
    "b12_folate_deficiency": {
        'Gender': 0,  # Female
        'Hemoglobin': 11.0,  # Low (anemic)
        'MCH': 31.0,  # Normal-high
        'MCHC': 33.0,  # Normal
        'MCV': 105.0,  # High (macrocytic)
        'RDW': 16.0,   # Elevated
        'RBC': 3.8,    # Low (typical in B12/folate deficiency)
        'Age': 68      # Elderly (B12 deficiency more common)
    },
    "thalassemia_trait": {
        'Gender': 1,  # Male
        'Hemoglobin': 13.0,  # Borderline
        'MCH': 24.0,  # Low
        'MCHC': 32.5,  # Normal
        'MCV': 74.0,   # Low (microcytic)
        'RDW': 13.5,   # Normal (key differentiator from iron deficiency)
        'RBC': 5.8,    # High (characteristic of thalassemia trait)
        'Age': 30      # Young adult
    },
    "anemia_chronic_disease": {
        'Gender': 0,  # Female
        'Hemoglobin': 11.5,  # Low (anemic)
        'MCH': 28.0,  # Normal
        'MCHC': 33.0,  # Normal
        'MCV': 86.0,   # Normal
        'RDW': 14.0,   # Normal
        'RBC': 3.9,    # Slightly low
        'Age': 72      # Elderly (ACD more common in older adults)
    }
}

def main():
    print("\n===== Early Anemia Risk Prediction Tool =====\n")
    print("Enter patient blood test values:")
    
    gender = input("Gender (1 for Male, 0 for Female): ").strip()
    hemoglobin = input("Hemoglobin (g/dL): ").strip()
    mch = input("MCH (pg): ").strip()
    mchc = input("MCHC (g/dL): ").strip()
    mcv = input("MCV (fL): ").strip()
    
    # Make additional parameters optional
    rdw = input("RDW (%) [Optional - press Enter to skip]: ").strip()
    rbc = input("RBC (10^6/μL) [Optional - press Enter to skip]: ").strip()
    age = input("Age (years) [Optional - press Enter to skip]: ").strip()
    
    patient_data = {
        'Gender': float(gender),
        'Hemoglobin': float(hemoglobin),
        'MCH': float(mch),
        'MCHC': float(mchc),
        'MCV': float(mcv)
    }
    
    # Add optional parameters if provided
    if rdw:
        patient_data['RDW'] = float(rdw)
    if rbc:
        patient_data['RBC'] = float(rbc)
    if age:
        patient_data['Age'] = float(age)
    
    # Include only features known to the model to avoid ValueError
    # We need to verify if the model was trained with these features
    try:
        # First, try with all features
        prediction, probability, category, analysis = predict_early_anemia_risk(
            patient_data, model, scaler, detailed=True
        )
    except ValueError as e:
        # If we get a feature names error, fall back to just the core features
        print("\nWarning: Some provided features weren't used in the original model. Using only core features.")
        core_patient_data = {
            'Gender': patient_data['Gender'],
            'Hemoglobin': patient_data['Hemoglobin'],
            'MCH': patient_data['MCH'],
            'MCHC': patient_data['MCHC'],
            'MCV': patient_data['MCV']
        }
        prediction, probability, category, analysis = predict_early_anemia_risk(
            core_patient_data, model, scaler, detailed=True
        )
        
        # Still include all parameters for reporting
        analysis['parameters'] = patient_data
    
    print_detailed_analysis(patient_data, prediction, probability, category, analysis)

if __name__ == "__main__":
    print("\n===== ANEMIA TYPE DETECTION & RISK PREDICTION =====\n")
    print("Choose an option:")
    print("1. Enter patient data manually")
    print("2. Test with Latent Iron Deficiency case")
    print("3. Test with Iron Deficiency Anemia case")
    print("4. Test with B12/Folate Deficiency case")
    print("5. Test with Thalassemia Trait case")
    print("6. Test with Anemia of Chronic Disease case")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        print("\nTesting with Latent Iron Deficiency case")
        prediction, probability, category, analysis = predict_early_anemia_risk(
            test_cases["latent_iron_deficiency"], model, scaler, detailed=True
        )
        print_detailed_analysis(test_cases["latent_iron_deficiency"], prediction, probability, category, analysis)
    elif choice == '3':
        print("\nTesting with Iron Deficiency Anemia case")
        prediction, probability, category, analysis = predict_early_anemia_risk(
            test_cases["iron_deficiency"], model, scaler, detailed=True
        )
        print_detailed_analysis(test_cases["iron_deficiency"], prediction, probability, category, analysis)
    elif choice == '4':
        print("\nTesting with B12/Folate Deficiency case")
        prediction, probability, category, analysis = predict_early_anemia_risk(
            test_cases["b12_folate_deficiency"], model, scaler, detailed=True
        )
        print_detailed_analysis(test_cases["b12_folate_deficiency"], prediction, probability, category, analysis)
    elif choice == '5':
        print("\nTesting with Thalassemia Trait case")
        prediction, probability, category, analysis = predict_early_anemia_risk(
            test_cases["thalassemia_trait"], model, scaler, detailed=True
        )
        print_detailed_analysis(test_cases["thalassemia_trait"], prediction, probability, category, analysis)
    elif choice == '6':
        print("\nTesting with Anemia of Chronic Disease case")
        prediction, probability, category, analysis = predict_early_anemia_risk(
            test_cases["anemia_chronic_disease"], model, scaler, detailed=True
        )
        print_detailed_analysis(test_cases["anemia_chronic_disease"], prediction, probability, category, analysis)
    else:
        print("Invalid choice. Please run the program again.")