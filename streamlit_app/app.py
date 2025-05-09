import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import py3Dmol
from stmol import showmol
import plotly.express as px
import plotly.graph_objects as go
from streamlit_shap import st_shap
import sys
import os

# Add the parent directory to path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.test_early_prediction import predict_early_anemia_risk, anemia_type_detector

# Page configuration
st.set_page_config(
    page_title="Anemia Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better appearance with medical color scheme (blues, whites, subtle reds)
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #0d3b66;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f2f7ff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #0d4c8f;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e6f3ff;
    }
    .sidebar .sidebar-content {
        background-color: #f2f7ff;
    }
    .stButton>button {
        background-color: #0d4c8f;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0a3d72;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    import os
    
    # Get the absolute path to the model directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model')
    
    # Load model and scaler with absolute paths
    model_path = os.path.join(model_dir, 'early_anemia_model.pkl')
    scaler_path = os.path.join(model_dir, 'early_anemia_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function for hemoglobin molecule visualization
def render_mol():
    pdb = """
    HEADER    HEMOGLOBIN
    COMPND    HEMOGLOBIN (DEOXY) CHAIN A
    ATOM      1  N   VAL A   1      -6.649  16.654  -0.835  1.00 15.16           N  
    ATOM      2  CA  VAL A   1      -7.454  15.520  -0.431  1.00 12.24           C  
    ATOM      3  C   VAL A   1      -8.039  15.691   0.947  1.00 14.12           C  
    ATOM      4  O   VAL A   1      -7.346  16.061   1.899  1.00 14.37           O  
    ATOM      5  CB  VAL A   1      -6.545  14.354  -0.472  1.00 14.00           C  
    ATOM      6  CG1 VAL A   1      -7.253  13.134  -0.035  1.00 11.36           C  
    ATOM      7  CG2 VAL A   1      -5.835  14.146  -1.778  1.00 12.91           C  
    HETATM  141  O   HOH A  31      -7.761   5.832  15.274  1.00 29.83           O  
    HETATM  142  FE  HEM A   1      -5.077  12.952   7.194  1.00  6.01          FE  
    HETATM  143  CHA HEM A   1      -7.100  14.607   5.877  1.00  6.48           C  
    HETATM  144  CHB HEM A   1      -3.682  15.029   7.789  1.00  6.10           C  
    HETATM  145  CHC HEM A   1      -3.198  11.320   8.782  1.00  6.01           C  
    HETATM  146  CHD HEM A   1      -6.612  10.975   6.865  1.00  5.37           C
    """
    viewer = py3Dmol.view(width=600, height=400)
    viewer.addModel(pdb, 'pdb')
    viewer.setStyle({'cartoon':{'color':'spectrum'}})
    viewer.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'redToBlue', 'radius':0.2}})
    viewer.addSurface(py3Dmol.VDW, {'opacity':0.6, 'color':'red'}, {'hetflag': {'list': ['HEM']}, 'invert': True})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    viewer.zoom(1.5)
    viewer.spin(True)
    showmol(viewer, height=400, width=600)

# Create SHAP explainer based on model type
@st.cache_resource
def create_shap_explainer(_model):
    try:
        # Check if the model is a tree-based model (RandomForest, GradientBoosting, etc.)
        if hasattr(_model, 'estimators_') or hasattr(_model, 'estimators'):
            return shap.TreeExplainer(_model)
        # If it's a linear model (LogisticRegression, LinearRegression, etc.)
        elif hasattr(_model, 'coef_'):
            # For logistic regression, create a KernelExplainer with a small background dataset
            background_data = shap.sample(pd.DataFrame(np.zeros((1, _model.coef_.shape[1])), 
                                      columns=[f'feature_{i}' for i in range(_model.coef_.shape[1])]), 5)
            return shap.KernelExplainer(_model.predict_proba, background_data)
        else:
            # Fallback to KernelExplainer for other model types
            background_data = pd.DataFrame(np.zeros((1, 10)))
            return shap.KernelExplainer(lambda x: x.mean(axis=1), background_data)
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {str(e)}")
        return None

# Navigation
def main():
    model, scaler = load_model()
    # Create SHAP explainer
    explainer = create_shap_explainer(model)
      # Sidebar navigation with improved design
    with st.sidebar:
        st.title("Anemia Prediction")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Hackathon", "Home", "Prediction", "Visualizations", "Educational Content", "About"],
            icons=["award", "house", "clipboard-check", "graph-up", "book", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f2f7ff"},
                "icon": {"color": "#0d4c8f"},
                "nav-link": {"color": "#0d3b66", "font-weight": "500"},
                "nav-link-selected": {"background-color": "#0d4c8f", "color": "white"},
            }
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This application helps predict anemia risk based on blood parameters. "
            "It uses machine learning to detect early signs even when values appear normal."
        )
      # Page content based on navigation
    if selected == "Hackathon":
        from landing_page import show_landing_page
        show_landing_page()
        
    elif selected == "Home":
        from home import show_home_page
        show_home_page()
        
    elif selected == "Prediction":
        st.title("Anemia Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("prediction_form"):
                st.subheader("Enter Patient Information")
                
                gender = st.radio("Gender", ["Male", "Female"])
                gender_val = 1 if gender == "Male" else 0
                
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1)
                mch = st.number_input("MCH (pg)", min_value=15.0, max_value=40.0, value=29.0, step=0.1)
                mchc = st.number_input("MCHC (g/dL)", min_value=25.0, max_value=40.0, value=33.0, step=0.1)
                mcv = st.number_input("MCV (fL)", min_value=60.0, max_value=120.0, value=89.0, step=0.1)
                rdw = st.number_input("RDW (%)", min_value=9.0, max_value=30.0, value=13.0, step=0.1)
                rbc = st.number_input("RBC (10^6/μL)", min_value=2.0, max_value=7.0, value=4.5, step=0.1)
                age = st.number_input("Age", min_value=1, max_value=100, value=35)
                
                submitted = st.form_submit_button("Predict", use_container_width=True)
        
        with col2:
            st.subheader("Normal Reference Ranges")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Hemoglobin (g/dL):**")
                st.markdown("- Male: 13.5-17.5")
                st.markdown("- Female: 12.0-15.5")
                st.markdown("**MCH (pg):** 27.0-33.0")
                st.markdown("**MCHC (g/dL):** 32.0-36.0")
            
            with col_b:
                st.markdown("**MCV (fL):** 80.0-100.0") 
                st.markdown("**RDW (%):** 11.5-14.5")
                st.markdown("**RBC (10^6/μL):**")
                st.markdown("- Male: 4.5-5.9")
                st.markdown("- Female: 4.0-5.2")
            
            # Display hemoglobin molecule visualization
            st.subheader("Hemoglobin Molecule")
            try:
                render_mol()
            except Exception as e:
                st.error(f"Failed to render 3D molecule: {str(e)}")
                st.info("Please make sure all necessary packages are installed: py3Dmol, stmol, and ipywidgets.")
                
        if submitted:
            # Prepare patient data
            patient_data = {
                'Gender': gender_val,
                'Hemoglobin': hemoglobin,
                'MCH': mch,
                'MCHC': mchc,
                'MCV': mcv,
                'RDW': rdw,
                'RBC': rbc,
                'Age': age
            }
            
            # Make prediction
            prediction, probability, category, analysis = predict_early_anemia_risk(
                patient_data, model, scaler, detailed=True
            )
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                # Risk assessment card with medical colors (blue for low risk, deeper red for high risk)
                # Ensure risk title matches the category and probability by deriving both from the probability value
                
                # Determine risk level and category consistently
                if probability < 0.3:
                    risk_label = "Low"
                    risk_category = "Low Risk"
                    risk_color = "#419D78"  # Green
                elif probability < 0.6:
                    risk_label = "Moderate" 
                    risk_category = "Moderate Risk"
                    risk_color = "#EF8354"  # Orange
                else:
                    risk_label = "High"
                    risk_category = "High Risk"
                    risk_color = "#EB5E55"  # Red
                
                # Use the consistent values in the display
                st.markdown(f"<div class='prediction-card' style='border-left: 6px solid {risk_color};'>"
                            f"<h3 style='color: {risk_color};'>"
                            f"{risk_label} Anemia Risk</h3>"
                            f"<p>Risk Probability: <strong>{probability:.2f}</strong></p>"
                            f"<p>Risk Category: <strong>{risk_category}</strong></p>"
                            f"</div>", unsafe_allow_html=True)
                
                # Anemia type assessment
                anemia_type = analysis['anemia_type']
                
                st.markdown(f"<div class='prediction-card'>"
                            f"<h3 style='color: #0d3b66;'>Anemia Type Assessment</h3>"
                            f"<p>Suspected Type: <strong>{anemia_type['type']}</strong></p>"
                            f"<p>Confidence: <strong>{anemia_type['confidence']:.2f}</strong></p>"
                            f"</div>", unsafe_allow_html=True)
                
                # Display recommendation
                if prediction == 1:
                    if probability >= 0.7:
                        recommendation = "Immediate clinical assessment recommended. Consider specialized blood tests."
                        rec_color = "#EB5E55"
                    else:
                        recommendation = "Follow-up within 3 months. Consider testing ferritin, iron, vitamin B12 levels."
                        rec_color = "#EF8354"
                else:
                    recommendation = "Routine follow-up recommended. No immediate action needed."
                    rec_color = "#419D78"
                
                st.markdown(f"<div class='prediction-card' style='border-left: 6px solid {rec_color};'>"
                            f"<h3 style='color: #0d3b66;'>Recommendation</h3>"
                            f"<p>{recommendation}</p>"
                            f"</div>", unsafe_allow_html=True)
            
            with col_result2:
                # Parameter status visualization
                st.subheader("Parameter Status")
                
                # Reference ranges
                gender_str = 'male' if gender_val == 1 else 'female'
                reference_ranges = {
                    'Hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 15.5)},
                    'MCH': (27.0, 33.0),
                    'MCHC': (32.0, 36.0),
                    'MCV': (80.0, 100.0),
                    'RDW': (11.5, 14.5),
                    'RBC': {'male': (4.5, 5.9), 'female': (4.0, 5.2)}
                }
                
                # Create individual gauge charts for each parameter
                params = ['Hemoglobin', 'MCH', 'MCHC', 'MCV', 'RDW', 'RBC']
                values = [hemoglobin, mch, mchc, mcv, rdw, rbc]
                
                # Create 2x3 grid for parameters
                param_cols = st.columns(2)
                
                for i, (param, value) in enumerate(zip(params, values)):
                    # Determine which column this parameter goes into
                    col_index = i % 2
                    
                    with param_cols[col_index]:
                        if param == 'Hemoglobin' or param == 'RBC':
                            ref_range = reference_ranges[param][gender_str]
                        else:
                            ref_range = reference_ranges[param]
                        
                        # Normalize to 0-1 range for the gauge
                        min_val = ref_range[0] - (ref_range[1] - ref_range[0]) * 0.5
                        max_val = ref_range[1] + (ref_range[1] - ref_range[0]) * 0.5
                        
                        # Create a single gauge chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Indicator(
                            mode = "gauge+number",
                            value = value,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': param, 'font': {'color': '#0d3b66', 'size': 16}},
                            gauge = {
                                'axis': {'range': [min_val, max_val], 'tickcolor': '#0d3b66'},
                                'bar': {'color': "#0d4c8f"},
                                'steps': [
                                    {'range': [min_val, ref_range[0]], 'color': "#F1D6C9"},  # Light orange/pink for low
                                    {'range': [ref_range[0], ref_range[1]], 'color': "#D3EBE9"},  # Light blue-green for normal
                                    {'range': [ref_range[1], max_val], 'color': "#F1D6C9"},  # Light orange/pink for high
                                ],
                                'threshold': {
                                    'line': {'color': "#0d3b66", 'width': 4},
                                    'thickness': 0.75,
                                    'value': value
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=200,
                            margin=dict(l=30, r=30, t=30, b=0),
                            paper_bgcolor='white',
                            font={'color': '#0d3b66'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Display patterns detected
            st.markdown("---")
            st.subheader("Blood Parameter Patterns")
            
            if 'patterns' in analysis and analysis['patterns']:
                for pattern in analysis['patterns']:
                    st.info(f"**{pattern['name']}**: {pattern['description']}")
            else:
                st.info("No specific anemia patterns detected")
            
            # Show SHAP analysis if we have an explainer
            st.markdown("---")
            st.subheader("Machine Learning Explanation")
            
            # Convert patient data to DataFrame for SHAP
            patient_df = pd.DataFrame([patient_data])
            
            # Add derived features that the model might expect
            if 'RBC' in patient_df.columns and 'MCV' in patient_df.columns:
                patient_df['RBC_MCV_Ratio'] = patient_df['RBC'] / patient_df['MCV']
            if 'RDW' in patient_df.columns and 'MCV' in patient_df.columns:
                patient_df['RDW_MCV_Ratio'] = patient_df['RDW'] / patient_df['MCV']
            if 'MCH' in patient_df.columns and 'MCHC' in patient_df.columns:
                patient_df['MCH_MCHC_Ratio'] = patient_df['MCH'] / patient_df['MCHC']
            if 'Age' in patient_df.columns and 'RBC' in patient_df.columns:
                patient_df['Age_RBC_Interaction'] = patient_df['Age'] * patient_df['RBC'] / 100
            
            # Extract feature_names_in_ from scaler if available
            if hasattr(scaler, 'feature_names_in_'):
                training_features = list(scaler.feature_names_in_)
            else:
                # Fallback to expected features
                training_features = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender',
                                'RBC_MCV_Ratio', 'RDW_MCV_Ratio', 'MCH_MCHC_Ratio',
                                'Age_RBC_Interaction', 'RDW_Age_Interaction']
            
            # Ensure the DataFrame has exactly the same features AND ORDER as used in training
            patient_df = patient_df.reindex(columns=training_features, fill_value=0)
            
            # Scale the features using the same scaler used during training
            scaled_patient = scaler.transform(patient_df)
            
            col_shap1, col_shap2 = st.columns(2)
            
            with col_shap1:
                st.write("### Feature Contribution")
                st.write("How each parameter influences the prediction:")
                
                try:
                    if explainer is not None:
                        # Different approach based on explainer type
                        if isinstance(explainer, shap.KernelExplainer):
                            # For KernelExplainer (e.g., for LogisticRegression)
                            shap_values = explainer.shap_values(scaled_patient)
                            
                            # For binary classification, shap_values is a list where second element is for positive class
                            if isinstance(shap_values, list) and len(shap_values) > 1:
                                # For binary classification (get positive class)
                                patient_shap_values = np.array(shap_values[1]).flatten()
                                # Get expected value safely
                                if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1:
                                    expected_value = explainer.expected_value[1]
                                else:
                                    expected_value = explainer.expected_value
                            else:
                                # For regression or single-output models
                                patient_shap_values = np.array(shap_values).flatten()
                                expected_value = explainer.expected_value
                            
                            # Create a simple horizontal bar chart for feature importance
                            plt.figure(figsize=(10, 6))
                            
                            # Make sure the number of features matches the number of SHAP values
                            feature_importance = []
                            for i, feature in enumerate(training_features):
                                if i < len(patient_shap_values):
                                    feature_importance.append((feature, patient_shap_values[i]))
                            
                            # Sort by absolute SHAP value
                            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Get top features and values (up to 10)
                            top_features = [x[0] for x in feature_importance[:min(10, len(feature_importance))]]
                            top_values = [x[1] for x in feature_importance[:min(10, len(feature_importance))]]
                            
                            # Create color map (positive = red, negative = blue)
                            colors = ['#EB5E55' if x > 0 else '#4285f4' for x in top_values]
                            
                            # Create horizontal bar chart
                            y_pos = np.arange(len(top_features))
                            plt.barh(y_pos, top_values, color=colors)
                            plt.yticks(y_pos, top_features)
                            plt.xlabel('SHAP Value (Impact on Prediction)')
                            plt.title('Feature Impact on Anemia Risk Prediction')
                            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                            
                            # Add base value annotation (safely convert to float)
                            try:
                                base_value = float(expected_value)
                            except:
                                base_value = 0.0
                            plt.figtext(0.1, 0.01, f"Base value: {base_value:.4f}", ha="left", fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            
                        elif isinstance(explainer, shap.TreeExplainer):
                            # Same approach for TreeExplainer
                            shap_values = explainer.shap_values(scaled_patient)
                            
                            if isinstance(shap_values, list) and len(shap_values) > 1:
                                # For binary classification (get positive class)
                                patient_shap_values = np.array(shap_values[1]).flatten()
                                # Get expected value safely
                                if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1:
                                    expected_value = explainer.expected_value[1]
                                else:
                                    expected_value = explainer.expected_value
                            else:
                                # For regression or single-output models
                                patient_shap_values = np.array(shap_values).flatten()
                                expected_value = explainer.expected_value
                            
                            # Create horizontal bar chart
                            plt.figure(figsize=(10, 6))
                            
                            # Make sure the number of features matches the number of SHAP values
                            feature_importance = []
                            for i, feature in enumerate(training_features):
                                if i < len(patient_shap_values):
                                    feature_importance.append((feature, patient_shap_values[i]))
                            
                            # Sort by absolute SHAP value
                            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Get top features and values (up to 10)
                            top_features = [x[0] for x in feature_importance[:min(10, len(feature_importance))]]
                            top_values = [x[1] for x in feature_importance[:min(10, len(feature_importance))]]
                            
                            # Create color map (positive = red, negative = blue)
                            colors = ['#EB5E55' if x > 0 else '#4285f4' for x in top_values]
                            
                            # Create horizontal bar chart
                            y_pos = np.arange(len(top_features))
                            plt.barh(y_pos, top_values, color=colors)
                            plt.yticks(y_pos, top_features)
                            plt.xlabel('SHAP Value (Impact on Prediction)')
                            plt.title('Feature Impact on Anemia Risk Prediction')
                            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                            
                            # Add base value annotation (safely convert to float)
                            try:
                                base_value = float(expected_value)
                            except:
                                base_value = 0.0
                            plt.figtext(0.1, 0.01, f"Base value: {base_value:.4f}", ha="left", fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                        else:
                            st.info("SHAP explainer type not supported for visualization")
                    else:
                        st.info("SHAP explainer not available")
                except Exception as e:
                    st.error(f"Error generating SHAP visualization: {str(e)}")
                    
                    # Fallback to basic feature importance display
                    st.write("Using model coefficients for feature importance:")
                    if hasattr(model, 'coef_'):
                        coefs = model.coef_[0]
                        feature_importance = sorted(zip(training_features, coefs), key=lambda x: abs(x[1]), reverse=True)
                        
                        # Create basic plot with model coefficients
                        plt.figure(figsize=(10, 6))
                        features = [x[0] for x in feature_importance[:10]]  # Top 10 features
                        importances = [x[1] for x in feature_importance[:10]]
                        colors = ['#4285f4' if imp < 0 else '#EB5E55' for imp in importances]
                        
                        plt.barh(range(len(features)), importances, color=colors)
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('Coefficient Value')
                        plt.title('Feature Importance from Model Coefficients')
                        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
            
            with col_shap2:
                st.write("### Feature Importance")
                st.write("The most important features for this prediction:")
                
                # Sort features by importance
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models
                    feature_importance = sorted(
                        zip(training_features, model.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    )
                elif hasattr(model, 'coef_'):
                    # For linear models
                    coefs = model.coef_[0]
                    feature_importance = sorted(
                        zip(training_features, [abs(c) for c in coefs]),
                        key=lambda x: x[1],
                        reverse=True
                    )
                else:
                    # Fallback using SHAP values if available
                    try:
                        if explainer is not None:
                            if isinstance(explainer, shap.KernelExplainer):
                                shap_values = explainer.shap_values(scaled_patient)
                                if isinstance(shap_values, list) and len(shap_values) > 1:
                                    patient_shap_values = shap_values[1][0]
                                else:
                                    patient_shap_values = shap_values[0]
                            else:
                                shap_values = explainer.shap_values(scaled_patient)
                                if isinstance(shap_values, list) and len(shap_values) > 1:
                                    patient_shap_values = shap_values[1][0]
                                else:
                                    patient_shap_values = shap_values[0]
                                    
                            feature_importance = sorted(
                                zip(training_features, [abs(v) for v in patient_shap_values]),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        else:
                            # If no feature importance available
                            feature_importance = [(feature, 1.0 / len(training_features)) for feature in training_features]
                    except Exception:
                        # If SHAP fails
                        feature_importance = [(feature, 1.0 / len(training_features)) for feature in training_features]
                
                # Create a bar chart of feature importance with medical color scheme
                feature_names = [x[0] for x in feature_importance[:10]]  # Show top 10 features
                feature_values = [x[1] for x in feature_importance[:10]]
                
                fig = plt.figure(figsize=(10, 6))
                bars = plt.barh(feature_names, feature_values, color='#0d4c8f')
                plt.xlabel('Feature Importance')
                plt.title('Top Features for Anemia Prediction')
                
                # Add value annotations
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                             f'{width:.4f}',
                             ha='left', va='center', color='#0d3b66')
                
                plt.tight_layout()
                st.pyplot(fig)

    # Rest of the code remains unchanged...
    elif selected == "Visualizations":
        st.title("Anemia Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Correlation Analysis", "Model Performance"])
        
        with tab1:
            st.subheader("Hemoglobin Distribution")
            st.image("../model/hemoglobin_distribution.png", use_container_width=True)
            st.markdown("""
                This plot shows the distribution of hemoglobin levels in the dataset. 
                Notice how there's a significant overlap between anemic and non-anemic populations, 
                which makes early detection challenging based on hemoglobin levels alone.
            """)
        
        with tab2:
            st.subheader("Parameter Correlations")
            st.image("../model/correlation_heatmap.png", use_container_width=True)
            st.markdown("""
                The heatmap shows correlations between different blood parameters. 
                Strong correlations indicate relationships that can help in early anemia detection 
                even when individual parameters are within normal ranges.
            """)
        
        with tab3:
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("../model/early_predict_cm_gradient_boosting.png", use_container_width=True)
                st.markdown("Confusion matrix showing model performance")
                
            with col2:
                st.image("../model/early_predict_importance.png", use_container_width=True)
                st.markdown("Feature importance in the prediction model")
            
            st.markdown("---")
            st.image("../model/roc_curves.png", use_container_width=True)
            st.markdown("""
                ROC curves showing the model's ability to distinguish between anemic and non-anemic cases.
                Higher area under the curve (AUC) indicates better model performance.
            """)
    
    elif selected == "Educational Content":
        st.title("Understanding Anemia")
        
        tab1, tab2, tab3 = st.tabs(["What is Anemia?", "Types of Anemia", "Interactive Blood Cell View"])
        
        with tab1:
            st.subheader("What is Anemia?")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                    **Anemia** is a condition characterized by a deficiency in the number or quality of red blood cells, 
                    resulting in reduced oxygen delivery to the body's tissues. 
                    
                    **Key facts about anemia:**
                    
                    - Affects over 1.62 billion people globally (about 25% of the population)
                    - More common in women, young children, and older adults
                    - Often goes undiagnosed in early stages
                    - Can significantly impact quality of life, causing fatigue, weakness, and reduced cognitive function
                    - Early detection can lead to simpler treatment and better outcomes
                    
                    **Key blood parameters in anemia diagnosis:**
                    
                    - **Hemoglobin (Hb)**: The protein in red blood cells that carries oxygen
                    - **Red Blood Cell Count (RBC)**: The number of red blood cells per volume of blood
                    - **Mean Corpuscular Volume (MCV)**: The average size of red blood cells
                    - **Mean Corpuscular Hemoglobin (MCH)**: The average amount of hemoglobin per red blood cell
                    - **Mean Corpuscular Hemoglobin Concentration (MCHC)**: The average concentration of hemoglobin in a given volume of red blood cells
                    - **Red Cell Distribution Width (RDW)**: A measure of the variation in red blood cell size
                """)
            
            with col2:
                render_mol()
                st.caption("3D model of hemoglobin molecule")
        
        with tab2:
            st.subheader("Types of Anemia")
            
            anemia_types = {
                "Iron Deficiency Anemia": {
                    "description": "The most common type of anemia, caused by insufficient iron to produce hemoglobin.",
                    "key_markers": "Low MCV, low MCH, high RDW, normal or low RBC",
                    "prevalence": "30% of the global population",
                },
                "B12/Folate Deficiency Anemia": {
                    "description": "Caused by insufficient vitamin B12 or folate, essential for producing healthy red blood cells.",
                    "key_markers": "High MCV, normal or low MCH, high RDW",
                    "prevalence": "Affects up to 15% of older adults",
                },
                "Thalassemia": {
                    "description": "Genetic disorders that affect hemoglobin production.",
                    "key_markers": "Low MCV, low MCH, normal RDW, often high RBC count",
                    "prevalence": "Carrier state in up to 5% in certain populations",
                },
                "Anemia of Chronic Disease": {
                    "description": "Associated with chronic inflammation, infections, or kidney disease.",
                    "key_markers": "Normal MCV, normal MCH, normal RDW, low hemoglobin",
                    "prevalence": "Second most common form of anemia globally",
                }
            }
            
            for anemia_type, info in anemia_types.items():
                with st.expander(anemia_type):
                    st.markdown(f"**Description**: {info['description']}")
                    st.markdown(f"**Key Markers**: {info['key_markers']}")
                    st.markdown(f"**Prevalence**: {info['prevalence']}")
        
        with tab3:
            st.subheader("Interactive Blood Cell Visualization")
            
            # Interactive red blood cell visualization using Plotly
            st.markdown("### Normal vs. Anemic Red Blood Cells")
            
            # Create a 3D scatter plot resembling blood cells
            def generate_sphere_points(center, radius, n=50):
                u = np.linspace(0, 2 * np.pi, n)
                v = np.linspace(0, np.pi, n)
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                return x, y, z
            
            # Create cells of different types
            fig = go.Figure()
            
            # Normal cell
            x, y, z = generate_sphere_points([0, 0, 0], 1)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'rgb(220,20,60)'], [1, 'rgb(178,34,34)']], 
                                     opacity=0.8, showscale=False, name="Normal RBC"))
            
            # Microcytic cell (smaller)
            x, y, z = generate_sphere_points([2.5, 0, 0], 0.7)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'rgb(205,92,92)'], [1, 'rgb(165,42,42)']], 
                                     opacity=0.8, showscale=False, name="Microcytic RBC"))
            
            # Macrocytic cell (larger)
            x, y, z = generate_sphere_points([-2.5, 0, 0], 1.3)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'rgb(250,128,114)'], [1, 'rgb(205,92,92)']], 
                                     opacity=0.8, showscale=False, name="Macrocytic RBC"))
            
            # Update the layout
            fig.update_layout(
                title="Red Blood Cell Types",
                scene=dict(
                    xaxis=dict(title="", showticklabels=False),
                    yaxis=dict(title="", showticklabels=False),
                    zaxis=dict(title="", showticklabels=False),
                ),
                legend=dict(x=0.7, y=0.1),
                width=800,
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
            
            # Add annotations
            fig.add_annotation(x=0, y=-0.15, text="Normal Red Blood Cell",
                              showarrow=False, xref="paper", yref="paper")
            
            fig.add_annotation(x=0.75, y=-0.15, text="Microcytic (Iron Deficiency)",
                              showarrow=False, xref="paper", yref="paper")
            
            fig.add_annotation(x=0.25, y=-0.15, text="Macrocytic (B12 Deficiency)",
                              showarrow=False, xref="paper", yref="paper")
            
            st.plotly_chart(fig)
            
            st.markdown("""
                #### Key Differences:
                
                - **Normal Red Blood Cells**: Biconcave disc shape with uniform size and color
                - **Microcytic Cells**: Smaller than normal, often seen in iron deficiency anemia
                - **Macrocytic Cells**: Larger than normal, often seen in B12 or folate deficiency
            """)
    
    elif selected == "About":
        st.title("About the Anemia Prediction System")
        
        st.markdown("""
        ### Project Overview
        
        This application uses machine learning to predict anemia risk based on standard blood parameters. What makes our approach unique is the ability to detect potential anemia risk even when all standard blood test values appear within normal clinical ranges.
        
        ### Features
        
        - **Early Detection**: Identify subclinical anemia before traditional diagnostic methods
        - **Risk Stratification**: Categorize patients into risk groups for appropriate follow-up
        - **Pattern Recognition**: Detect subtle patterns in blood parameters
        - **Explainable AI**: Use SHAP values to explain predictions
        - **Personalized Recommendations**: Suggest follow-up tests based on suspected anemia type
        
        ### Technology Stack
        
        - **Machine Learning**: Scikit-learn, SHAP for explainability
        - **Data Visualization**: Matplotlib, Seaborn, Plotly
        - **Web Interface**: Streamlit
        - **3D Visualization**: py3Dmol for molecular visualization
        
        ### Team
        
        This project was developed by the Health Analytics Research Team as part of an initiative to improve early diagnosis of common medical conditions using machine learning.
        """)

if __name__ == "__main__":
    main()