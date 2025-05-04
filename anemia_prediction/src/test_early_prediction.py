import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Load the saved model and scaler
model = joblib.load('model/early_anemia_model.pkl')
scaler = joblib.load('model/early_anemia_scaler.pkl')

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

def perform_shap_analysis(patient_data, model, scaler):
    """
    Perform SHAP analysis to explain the prediction for this patient
    
    Parameters:
    patient_data: dict with patient's blood parameters
    model: trained model
    scaler: fitted scaler
    
    Returns:
    shap_values: SHAP values for the prediction
    feature_names: names of the features
    base_value: baseline value of the model
    """
    # Create a copy of patient data for model input
    model_input_data = {k: v for k, v in patient_data.items()}
    
    # Create a DataFrame for the patient with derived features
    patient_df = pd.DataFrame([model_input_data])
    
    # Create original derived features
    patient_df['Hb_MCH_Ratio'] = patient_df['Hemoglobin'] / patient_df['MCH']
    patient_df['Hb_MCHC_Ratio'] = patient_df['Hemoglobin'] / patient_df['MCHC'] 
    patient_df['MCH_MCV_Ratio'] = patient_df['MCH'] / patient_df['MCV']
    patient_df['MCHC_MCV_Ratio'] = patient_df['MCHC'] / patient_df['MCV']
    patient_df['Hb_Gender_Interaction'] = patient_df['Hemoglobin'] * patient_df['Gender']
    
    # Create new derived features for RBC, RDW, and Age if available
    if 'RBC' in patient_data:
        patient_df['RBC_Hb_Ratio'] = patient_df['RBC'] / patient_df['Hemoglobin']
    
    if 'RDW' in patient_data and 'MCV' in patient_data:
        patient_df['RDW_MCV_Ratio'] = patient_df['RDW'] / patient_df['MCV']
    
    if 'Age' in patient_data and 'Hemoglobin' in patient_data:
        patient_df['Age_Hb_Interaction'] = patient_df['Age'] * patient_df['Hemoglobin'] / 100
    
    if 'Age' in patient_data and 'RBC' in patient_data:
        patient_df['RBC_Age_Interaction'] = patient_df['RBC'] * np.log(patient_df['Age'] + 1)
    
    # Scale the features while preserving column names
    feature_names = patient_df.columns
    patient_scaled = pd.DataFrame(
        scaler.transform(patient_df),
        columns=feature_names
    )
    
    # Create background data for SHAP (use a sample of typical values)
    # In a real implementation, we'd use a representative sample of the dataset
    typical_values = {}
    typical_values['male'] = pd.DataFrame([{
        'Gender': 1,
        'Hemoglobin': 15.0,
        'MCH': 30.0,
        'MCHC': 34.0,
        'MCV': 90.0,
        'Hb_MCH_Ratio': 15.0/30.0,
        'Hb_MCHC_Ratio': 15.0/34.0,
        'MCH_MCV_Ratio': 30.0/90.0,
        'MCHC_MCV_Ratio': 34.0/90.0,
        'Hb_Gender_Interaction': 15.0*1,
        'RBC': 5.2,
        'RDW': 13.0,
        'Age': 45,
        'RBC_Hb_Ratio': 5.2/15.0,
        'RDW_MCV_Ratio': 13.0/90.0,
        'Age_Hb_Interaction': 45*15.0/100,
        'RBC_Age_Interaction': 5.2*np.log(45+1)
    }])
    typical_values['female'] = pd.DataFrame([{
        'Gender': 0,
        'Hemoglobin': 13.5,
        'MCH': 30.0,
        'MCHC': 34.0,
        'MCV': 90.0,
        'Hb_MCH_Ratio': 13.5/30.0,
        'Hb_MCHC_Ratio': 13.5/34.0,
        'MCH_MCV_Ratio': 30.0/90.0,
        'MCHC_MCV_Ratio': 34.0/90.0,
        'Hb_Gender_Interaction': 13.5*0,
        'RBC': 4.6,
        'RDW': 13.0,
        'Age': 45,
        'RBC_Hb_Ratio': 4.6/13.5,
        'RDW_MCV_Ratio': 13.0/90.0,
        'Age_Hb_Interaction': 45*13.5/100,
        'RBC_Age_Interaction': 4.6*np.log(45+1)
    }])
    
    # Choose appropriate background data based on gender
    gender = 'male' if patient_data['Gender'] == 1 else 'female'
    background_data = typical_values[gender]
    
    # Filter background data to include only columns present in patient_scaled
    background_data = background_data[[col for col in background_data.columns if col in patient_scaled.columns]]
    
    # Create SHAP explainer with check_additivity=False to suppress additivity warnings
    if hasattr(model, 'predict_proba'):
        explainer = shap.Explainer(model, background_data)
        shap_values = explainer(patient_scaled, check_additivity=False)
        # SHAP values for the probability of class 1 (anemia)
        try:
            return shap_values[:, :, 1], feature_names, explainer.expected_value[1]
        except:
            # For some SHAP versions the format is different
            return shap_values, feature_names, explainer.expected_value
    else:
        explainer = shap.Explainer(model, background_data)
        shap_values = explainer(patient_scaled, check_additivity=False)
        return shap_values, feature_names, explainer.expected_value

def print_shap_analysis(shap_values, feature_names, base_value, patient_data):
    """
    Print and visualize SHAP analysis results
    """
    print("\n----- SHAP FEATURE CONTRIBUTION ANALYSIS -----")
    
    # Get SHAP values as a dictionary for easier access
    try:
        # For newer SHAP versions
        feature_contributions = dict(zip(feature_names, shap_values[0].values))
    except:
        # Fall back for different SHAP versions
        feature_contributions = dict(zip(feature_names, shap_values[0]))
    
    # Sort features by absolute SHAP value
    sorted_features = sorted(
        feature_contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    print(f"Baseline risk: {base_value:.4f}")
    print("Top feature contributions (positive values increase risk, negative values decrease risk):")
    
    for feature, value in sorted_features[:5]:  # Show top 5 features
        direction = "INCREASES" if value > 0 else "DECREASES"
        print(f"• {feature}: {value:.4f} ({direction} risk)")
    
    # Create a SHAP waterfall plot for the patient
    plt.figure(figsize=(10, 6))
    try:
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title("SHAP Feature Contributions")
        plt.tight_layout()
        plt.savefig('model/shap_waterfall.png')
        print("\nSHAP waterfall plot saved to model/shap_waterfall.png")
    except:
        print("\nCouldn't create waterfall plot - may need updated SHAP version")
    
    # Return the sorted features for use in the detailed analysis
    return sorted_features

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
    
    # Create the basic derived features
    patient_df['Hb_MCH_Ratio'] = patient_df['Hemoglobin'] / patient_df['MCH']
    patient_df['Hb_MCHC_Ratio'] = patient_df['Hemoglobin'] / patient_df['MCHC'] 
    patient_df['MCH_MCV_Ratio'] = patient_df['MCH'] / patient_df['MCV']
    patient_df['MCHC_MCV_Ratio'] = patient_df['MCHC'] / patient_df['MCV']
    patient_df['Hb_Gender_Interaction'] = patient_df['Hemoglobin'] * patient_df['Gender']
    
    # Create derived features for the new parameters if available
    if 'RBC' in patient_data:
        patient_df['RBC_Hb_Ratio'] = patient_df['RBC'] / patient_df['Hemoglobin']
        
    if 'RDW' in patient_data and 'MCV' in patient_data:
        patient_df['RDW_MCV_Ratio'] = patient_df['RDW'] / patient_df['MCV']
        
    if 'Age' in patient_data and 'Hemoglobin' in patient_data:
        patient_df['Age_Hb_Interaction'] = patient_df['Age'] * patient_df['Hemoglobin'] / 100
        
    if 'Age' in patient_data and 'RBC' in patient_data:
        patient_df['RBC_Age_Interaction'] = patient_df['RBC'] * np.log(patient_df['Age'] + 1)
    
    # Scale the features
    try:
        patient_scaled = pd.DataFrame(
            scaler.transform(patient_df),
            columns=patient_df.columns
        )
    except ValueError as e:
        # If we get a feature mismatch error, print more details
        if "feature_names_in_" in str(e) or "feature names" in str(e):
            if hasattr(scaler, 'feature_names_in_'):
                missing_features = set(scaler.feature_names_in_) - set(patient_df.columns)
                extra_features = set(patient_df.columns) - set(scaler.feature_names_in_)
                if missing_features:
                    print(f"Missing features from model training: {missing_features}")
                if extra_features:
                    print(f"Extra features not in model: {extra_features}")
                # Use only the features that were available during model training
                patient_df = patient_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
            else:
                print("Scaler doesn't have feature_names_in_ attribute. Using core features only.")
                # Fall back to core features only
                patient_df = pd.DataFrame([{
                    'Gender': patient_data['Gender'],
                    'Hemoglobin': patient_data['Hemoglobin'],
                    'MCH': patient_data['MCH'],
                    'MCHC': patient_data['MCHC'],
                    'MCV': patient_data['MCV']
                }])
            # Try again with corrected features
            patient_scaled = pd.DataFrame(
                scaler.transform(patient_df),
                columns=patient_df.columns
            )
        else:
            raise  # Re-raise if it's not a feature mismatch error
    
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
    
    # If detailed analysis is requested, add SHAP values
    if detailed:
        try:
            # Perform SHAP analysis
            shap_values, feature_names, base_value = perform_shap_analysis(patient_data, model, scaler)
            analysis['shap'] = {
                'values': shap_values,
                'feature_names': feature_names,
                'base_value': base_value
            }
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
    
    return adjusted_prediction, adjusted_probability, risk_category, analysis

def print_detailed_analysis(patient_data, prediction=None, probability=None, category=None, analysis=None):
    """Print a detailed clinical analysis of the blood parameters"""
    if analysis is None:
        _, _, _, analysis = predict_early_anemia_risk(patient_data, model, scaler, detailed=True)
    
    gender = "Male" if patient_data['Gender'] == 1 else "Female"
    
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
    
    # Add SHAP analysis to the output if available
    if 'shap' in analysis:
        try:
            sorted_features = print_shap_analysis(
                analysis['shap']['values'], 
                analysis['shap']['feature_names'], 
                analysis['shap']['base_value'],
                patient_data
            )
            
            # Enhance the conclusion with SHAP insights
            print("\n----- FEATURE IMPORTANCE INSIGHTS -----")
            for feature, value in sorted_features[:3]:  # Focus on top 3
                if 'Hemoglobin' in feature and value < 0:
                    print(f"• High hemoglobin ({patient_data['Hemoglobin']}) significantly reduces anemia risk")
                elif 'Hemoglobin' in feature and value > 0:
                    print(f"• Low hemoglobin ({patient_data['Hemoglobin']}) significantly increases anemia risk")
                elif 'MCH' in feature and value > 0:
                    print(f"• Low MCH ({patient_data['MCH']}) contributes to increased risk")
                elif 'MCHC' in feature and value > 0:
                    print(f"• Low MCHC ({patient_data['MCHC']}) suggests hypochromic pattern")
                elif 'MCV' in feature and value > 0 and patient_data['MCV'] < 80:
                    print(f"• Low MCV ({patient_data['MCV']}) indicates microcytic pattern")
                elif 'MCV' in feature and value > 0 and patient_data['MCV'] > 100:
                    print(f"• High MCV ({patient_data['MCV']}) indicates macrocytic pattern")
                elif 'Ratio' in feature and value > 0:
                    print(f"• Abnormal {feature} suggests imbalance in red cell parameters")
        except Exception as e:
            print(f"\nSHAP analysis display failed: {str(e)}")
    
    print("\n==================================================")

# Add test cases specifically for different anemia types
test_cases = {
    "iron_deficiency": {
        'Gender': 1,  # Male
        'Hemoglobin': 12.0,  # Low (anemic)
        'MCH': 22.0,  # Low (hypochromic)
        'MCHC': 29.0,  # Low (hypochromic)
        'MCV': 75.0,   # Low (microcytic)
        'RDW': 16.5    # Elevated (typical in iron deficiency)
    },
    "latent_iron_deficiency": {
        'Gender': 1,  # Male
        'Hemoglobin': 16.0,  # Normal/high
        'MCH': 22.0,  # Low (hypochromic)
        'MCHC': 29.0,  # Low (hypochromic)
        'MCV': 82.0,   # Low-normal
        'RDW': 15.2    # Slightly elevated (early changes)
    },
    "b12_folate_deficiency": {
        'Gender': 0,  # Female
        'Hemoglobin': 11.0,  # Low (anemic)
        'MCH': 31.0,  # Normal-high
        'MCHC': 33.0,  # Normal
        'MCV': 105.0,  # High (macrocytic)
        'RDW': 16.0    # Elevated
    },
    "thalassemia_trait": {
        'Gender': 1,  # Male
        'Hemoglobin': 13.0,  # Borderline
        'MCH': 24.0,  # Low
        'MCHC': 32.5,  # Normal
        'MCV': 74.0,   # Low (microcytic)
        'RDW': 13.5    # Normal (key differentiator from iron deficiency)
    },
    "anemia_chronic_disease": {
        'Gender': 0,  # Female
        'Hemoglobin': 11.5,  # Low (anemic)
        'MCH': 28.0,  # Normal
        'MCHC': 33.0,  # Normal
        'MCV': 86.0,   # Normal
        'RDW': 14.0    # Normal
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