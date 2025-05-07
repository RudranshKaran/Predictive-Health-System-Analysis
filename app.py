# Entry point for the Streamlit dashboard

import pandas as pd
import streamlit as st
from src.patient_summary import PatientSummary
from src.biomarker_analysis import BiomarkerAnalysis
from src.regional_analysis import RegionalAnalysis

# Initialize the modules

@st.cache_resource
def init_modules():
    patient_module = PatientSummary()
    biomarker_module = BiomarkerAnalysis()
    regional_module = RegionalAnalysis()
    
    # Load data
    patient_module.load_data("data/patient_data.csv")
    biomarker_module.load_data("data/biomarker_data.csv", "data/reference_ranges.csv")
    regional_module.load_data("data/regional_data.csv")
    
    return patient_module, biomarker_module, regional_module

def main():
    st.title("Predictive Health System Analysis")
    
    # Initialize modules
    try:
        patient_module, biomarker_module, regional_module = init_modules()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar navigation
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Patient Summary", "Biomarker Analysis", "Regional Trends", "Model Performance"]
    )
    
    if analysis_type == "Patient Summary":
        st.header("Patient Summary Analysis")
        
        # Patient ID input
        patient_id = st.text_input("Enter Patient ID")
        
        if patient_id:
            # Get patient timeline
            timeline = patient_module.get_visit_timeline(patient_id)
            
            if timeline:
                st.subheader("Visit Timeline")
                st.plotly_chart(patient_module.plot_visit_history(patient_id))
                
                st.subheader("Recurring Illnesses")
                recurring = patient_module.get_recurring_illnesses(patient_id)
                if recurring:
                    for illness in recurring:
                        st.write(f"- {illness}")
                else:
                    st.write("No recurring illnesses found")
            else:
                st.warning("No data found for this patient ID")
                
    elif analysis_type == "Biomarker Analysis":
        st.header("Biomarker Analysis")
        
        patient_id = st.text_input("Enter Patient ID")
        if patient_id:
            analysis = biomarker_module.analyze_biomarkers(patient_id)
            
            if analysis:
                # Display abnormal markers
                st.subheader("Abnormal Biomarkers")
                if analysis['abnormal_markers']:
                    for marker in analysis['abnormal_markers']:
                        st.warning(
                            f"{marker['marker']}: {marker['value']} "
                            f"(Reference Range: {marker['reference_range']})"
                        )
                else:
                    st.success("All biomarkers within normal ranges")
                
                # Display trends
                st.subheader("Biomarker Trends")
                for marker, trend in analysis['trend_analysis'].items():
                    st.write(
                        f"- {marker}: {trend['trend'].title()} "
                        f"(magnitude: {trend['magnitude']:.2f})"
                    )
                
                # Display risk factors
                if analysis['risk_factors']:
                    st.subheader("Identified Risk Factors")
                    for risk in analysis['risk_factors']:
                        st.error(f"- {risk}")
            else:
                st.warning("No biomarker data found for this patient ID")
                
    elif analysis_type == "Regional Trends":
        st.header("Regional Health Trends")
        
        region = st.selectbox(
            "Select Region",
            ["North", "South", "East", "West", "Central"]
        )
        
        if region:
            # Get regional analysis
            analysis = regional_module.analyze_regional_patterns(region)
            
            if analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display disease prevalence
                    st.subheader("Disease Prevalence")
                    for disease, stats in analysis['disease_prevalence'].items():
                        st.write(
                            f"- {disease}: {stats['percentage']:.1f}% "
                            f"({stats['count']} cases)"
                        )
                        
                        # Check for hotspots for each disease
                        hotspots = regional_module.identify_hotspots(disease)
                        if hotspots:
                            for hotspot in hotspots:
                                if hotspot['severity'] == 'high':
                                    st.error(f"⚠️ High-severity cluster detected: {hotspot['count']} cases, {hotspot['recent_cases']} in last 14 days")
                                elif hotspot['severity'] == 'medium':
                                    st.warning(f"⚠️ Medium-severity cluster detected: {hotspot['count']} cases, {hotspot['recent_cases']} in last 14 days")
                
                with col2:
                    # Display health indicators
                    st.subheader("Health Indicators")
                    for indicator, stats in analysis['health_indicators'].items():
                        st.write(
                            f"- {indicator}: "
                            f"Mean: {stats['mean']:.2f}, "
                            f"Std: {stats['std']:.2f}"
                        )
                
                # Demographics Analysis
                st.subheader("Demographic Patterns")
                if 'demographic_patterns' in analysis:
                    demographics = analysis['demographic_patterns']
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.write("Age Distribution")
                        for age_group, count in demographics['age_groups'].items():
                            st.write(f"- {age_group}: {count}")
                    
                    with col4:
                        st.write("Gender Distribution")
                        for gender, count in demographics['gender_distribution'].items():
                            st.write(f"- {gender}: {count}")
                
                # Show heatmap
                st.subheader("Regional Health Heatmap")
                indicator = st.selectbox(
                    "Select Health Indicator",
                    list(analysis['health_indicators'].keys())
                )
                if indicator:
                    heatmap = regional_module.plot_regional_heatmap(indicator)
                    st.plotly_chart(heatmap)
            else:
                st.warning("No data available for selected region")
                
    else:  # Model Performance
        st.header("Model Performance Metrics")
        
        # Add options for model configuration
        st.sidebar.subheader("Model Configuration")
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100, 10)
        
        if st.button("Calculate Performance Metrics"):
            with st.spinner("Training model and calculating metrics..."):
                # Get performance metrics with configured parameters
                metrics = biomarker_module.train_and_evaluate_model(test_size=test_size, n_estimators=n_estimators)
                
                if metrics:
                    # Add model summary
                    st.markdown("### Model Summary")
                    st.info(f"""
                    - Model Type: Random Forest Classifier
                    - Number of Trees: {n_estimators}
                    - Test Set Size: {test_size*100}%
                    - Features: Cholesterol, Blood Pressure, Glucose, Hemoglobin, White Blood Cells
                    """)
                    
                    # Display metrics in tabs
                    tab1, tab2, tab3 = st.tabs(["Overall Metrics", "Detailed Report", "Confusion Matrix"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Macro F1", f"{metrics['f1_scores']['macro']:.3f}")
                            st.metric("Macro Recall", f"{metrics['recall_scores']['macro']:.3f}")
                        with col2:
                            st.metric("Weighted F1", f"{metrics['f1_scores']['weighted']:.3f}")
                            st.metric("Weighted Recall", f"{metrics['recall_scores']['weighted']:.3f}")
                            
                        if 'accuracy' in metrics:
                            st.metric("Overall Accuracy", f"{metrics['accuracy']:.3f}")
                    
                    with tab2:
                        st.subheader("Classification Report")
                        st.text(metrics['classification_report'])
                        
                        if 'feature_importance' in metrics:
                            st.subheader("Feature Importance")
                            feature_imp = pd.DataFrame(
                                metrics['feature_importance'],
                                columns=['Importance']
                            ).sort_values('Importance', ascending=True)
                            st.bar_chart(feature_imp)
                    
                    with tab3:
                        st.subheader("Confusion Matrix")
                        fig = biomarker_module.plot_confusion_matrix(metrics)
                        if fig:
                            st.pyplot(fig)
                            
                        # Add confusion matrix interpretation
                        st.markdown("### Matrix Interpretation")
                        st.info("""
                        The confusion matrix shows:
                        - True Positives (diagonal): Correctly predicted cases
                        - False Positives (columns): Incorrectly predicted positives
                        - False Negatives (rows): Missed cases
                        """)
                else:
                    st.error("No data available to calculate metrics")

if __name__ == "__main__":
    main()
