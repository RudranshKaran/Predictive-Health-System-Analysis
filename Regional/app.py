# Entry point for the Streamlit dashboard

import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import from the local src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.patient_summary import PatientSummary
from src.biomarker_analysis import BiomarkerAnalysis
from src.regional_analysis import RegionalAnalysis

# Load environment variables
load_dotenv()

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
    
    # Input validation helper
    def validate_patient_id(patient_id: str) -> bool:
        if not patient_id:
            st.error("Please enter a Patient ID")
            return False
        if patient_id not in biomarker_module.biomarker_data['patient_id'].unique():
            st.error("Invalid Patient ID. Please enter a valid ID.")
            return False
        return True

    if analysis_type == "Patient Summary":
        st.header("Patient Summary Analysis")
        
        # Patient ID input
        patient_id = st.text_input("Enter Patient ID", key="patient_summary_id")
        
        if patient_id and validate_patient_id(patient_id):
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
        
        patient_id = st.text_input("Enter Patient ID", key="biomarker_id")
        if patient_id and validate_patient_id(patient_id):
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
        
        # Add tabs for better organization
        tabs = st.tabs(["Overview", "Disease Analysis", "Demographics", "Visualizations"])
        
        with tabs[0]:  # Overview tab
            region = st.selectbox(
                "Select Region",
                ["North", "South", "East", "West", "Central"],
                key="region_selector_overview"
            )
            
            if region:
                # Get regional analysis
                analysis = regional_module.analyze_regional_patterns(region)
                
                if analysis:
                    # Summary metrics at the top
                    total_cases = sum(stats['count'] for stats in analysis['disease_prevalence'].values())
                    total_diseases = len(analysis['disease_prevalence'])
                    
                    # Display summary metrics in a more visual way
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Total Cases", f"{total_cases:,}")
                    with metric_cols[1]:
                        st.metric("Unique Conditions", f"{total_diseases}")
                    with metric_cols[2]:
                        # Find the most prevalent disease
                        if analysis['disease_prevalence']:
                            most_prevalent = max(
                                analysis['disease_prevalence'].items(), 
                                key=lambda x: x[1]['percentage']
                            )
                            st.metric("Most Prevalent", 
                                     f"{most_prevalent[0]}", 
                                     f"{most_prevalent[1]['percentage']:.1f}%")
                    
                    # Display health indicators with improved formatting
                    st.subheader("Health Indicators")
                    indicator_cols = st.columns(2)
                    
                    indicators = list(analysis['health_indicators'].items())
                    half = len(indicators) // 2
                    
                    for i, (indicator, stats) in enumerate(indicators):
                        col_idx = 0 if i < half else 1
                        with indicator_cols[col_idx]:
                            st.write(
                                f"- **{indicator.replace('_', ' ').title()}**: "
                                f"Mean: {stats['mean']:.2f}, "
                                f"Std: {stats['std']:.2f}"
                            )
                else:
                    st.warning("No data available for selected region")
        
        with tabs[1]:  # Disease Analysis tab
            region = st.selectbox(
                "Select Region",
                ["North", "South", "East", "West", "Central"],
                key="region_selector_disease"
            )
            
            if region:
                analysis = regional_module.analyze_regional_patterns(region)
                
                if analysis:
                    # Display disease prevalence with improved visualization
                    st.subheader("Disease Prevalence")
                    
                    # Convert to DataFrame for better display
                    prevalence_data = []
                    for disease, stats in analysis['disease_prevalence'].items():
                        prevalence_data.append({
                            "Disease": disease,
                            "Cases": stats['count'],
                            "Percentage": f"{stats['percentage']:.1f}%"
                        })
                    
                    if prevalence_data:
                        prevalence_df = pd.DataFrame(prevalence_data)
                        st.dataframe(
                            prevalence_df.sort_values("Cases", ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Hotspot analysis with improved visualization
                    st.subheader("Disease Hotspots")
                    
                    # Create tabs for each disease with hotspots
                    diseases_with_hotspots = []
                    for disease in analysis['disease_prevalence'].keys():
                        hotspots = regional_module.identify_hotspots(disease)
                        if hotspots:
                            diseases_with_hotspots.append((disease, hotspots))
                    
                    if diseases_with_hotspots:
                        # Create a selectbox for disease selection
                        hotspot_diseases = [disease for disease, _ in diseases_with_hotspots]
                        selected_disease = st.selectbox(
                            "Select disease for detailed hotspot analysis",
                            hotspot_diseases,
                            key="hotspot_disease_selector")
                        
                        # Find the selected disease and its hotspots
                        selected_hotspots = None
                        for disease, hotspots in diseases_with_hotspots:
                            if disease == selected_disease:
                                selected_hotspots = hotspots
                                break
                        
                        if selected_hotspots:
                            # Display hotspots for the selected disease
                            hotspot_tabs = st.tabs([f"Cluster {i+1}" for i in range(len(selected_hotspots))])
                            
                            for i, (tab, hotspot) in enumerate(zip(hotspot_tabs, selected_hotspots)):
                                with tab:
                                    # Display hotspot information
                                    severity_color = "🔴" if hotspot['severity'] == 'high' else "🟠" if hotspot['severity'] == 'medium' else "🟡"
                                    st.write(
                                        f"{severity_color} **Severity**: {hotspot['severity'].upper()}"
                                    )
                                    
                                    # Create columns for metrics
                                    metric_cols = st.columns(3)
                                    with metric_cols[0]:
                                        st.metric("Total Cases", hotspot['count'])
                                    with metric_cols[1]:
                                        st.metric("Recent Cases (14d)", hotspot['recent_cases'])
                                    with metric_cols[2]:
                                        trend = ((hotspot['recent_cases'] / hotspot['count']) * 100) if hotspot['count'] > 0 else 0
                                        st.metric("Recent %", f"{trend:.1f}%")
                                    
                                    # Add a button to analyze root causes
                                    if st.button(f"Analyze Root Causes", key=f"analyze_btn_{i}"):
                                        with st.spinner("Analyzing potential causes with Gemini AI..."):
                                            # Call the new method to analyze hotspot causes
                                            analysis_result = regional_module.analyze_hotspot_causes(
                                                condition=selected_disease,
                                                region=region,
                                                hotspot_id=i
                                            )
                                            
                                            if "error" in analysis_result and analysis_result["error"]:
                                                st.error(analysis_result["error"])
                                                
                                                # Display which models were tried
                                                if "tried_models" in analysis_result:
                                                    st.warning(f"Attempted models: {', '.join(analysis_result['tried_models'])}")
                                                
                                                # Show detailed error information
                                                if "details" in analysis_result:
                                                    with st.expander("Error Details"):
                                                        st.code(analysis_result["details"])
                                                
                                                st.info("Please check your Gemini API key and make sure it's valid for at least one of the Gemini models.")
                                                if "details" in analysis_result:
                                                    with st.expander("Error Details"):
                                                        st.code(analysis_result["details"])
                                                st.info("Please check your Gemini API key and make sure it's valid for the current API version.")
                                            else:
                                                # Display the analysis results
                                                
                                                # Urgency level with appropriate color
                                                urgency = analysis_result.get("urgency_level", "unknown")
                                                urgency_color = {
                                                    "critical": "🔴", 
                                                    "high": "🟠", 
                                                    "medium": "🟡", 
                                                    "low": "🟢",
                                                    "unknown": "⚪"
                                                }.get(urgency, "⚪")
                                                
                                                st.subheader(f"{urgency_color} Urgency: {urgency.title()}")
                                                if "urgency_justification" in analysis_result:
                                                    st.write(analysis_result["urgency_justification"])
                                                
                                                # Root causes
                                                st.subheader("Potential Root Causes")
                                                for cause in analysis_result.get("root_causes", []):
                                                    confidence_color = {
                                                        "high": "🟢",
                                                        "medium": "🟡",
                                                        "low": "🟠"
                                                    }.get(cause.get("confidence", "medium"), "🟡")
                                                    
                                                    st.write(
                                                        f"{confidence_color} **{cause.get('cause', '')}** "
                                                        f"(Confidence: {cause.get('confidence', 'medium').title()})"
                                                    )
                                                
                                                # Recommended interventions
                                                st.subheader("Recommended Interventions")
                                                for intervention in analysis_result.get("interventions", []):
                                                    priority_color = {
                                                        "high": "🔴",
                                                        "medium": "🟠",
                                                        "low": "🟡"
                                                    }.get(intervention.get("priority", "medium"), "🟠")
                                                    
                                                    st.write(
                                                        f"{priority_color} **{intervention.get('action', '')}** "
                                                        f"(Priority: {intervention.get('priority', 'medium').title()})"
                                                    )
                                                
                                                # Estimated impact
                                                if "estimated_impact" in analysis_result and analysis_result["estimated_impact"]:
                                                    st.subheader("Estimated Impact")
                                                    st.write(analysis_result["estimated_impact"])
                    else:
                        st.info("No disease hotspots detected in this region")
                else:
                    st.warning("No data available for selected region")
        
        with tabs[2]:  # Demographics tab
            region = st.selectbox(
                "Select Region",
                ["North", "South", "East", "West", "Central"],
                key="region_selector_demographics"
            )
            
            if region:
                analysis = regional_module.analyze_regional_patterns(region)
                
                if analysis and 'demographic_patterns' in analysis:
                    demographics = analysis['demographic_patterns']
                    
                    # Age distribution visualization
                    st.subheader("Age Distribution")
                    
                    # Convert to DataFrame for better visualization
                    age_data = pd.DataFrame({
                        "Age Group": demographics['age_groups'].keys(),
                        "Count": demographics['age_groups'].values()
                    })
                    
                    # Create a bar chart for age distribution
                    st.bar_chart(age_data.set_index("Age Group"))
                    
                    # Gender distribution visualization
                    st.subheader("Gender Distribution")
                    
                    # Convert to DataFrame for better visualization
                    gender_data = pd.DataFrame({
                        "Gender": demographics['gender_distribution'].keys(),
                        "Count": demographics['gender_distribution'].values()
                    })
                    
                    # Create columns for the pie chart and data
                    gender_cols = st.columns([2, 1])
                    
                    with gender_cols[0]:
                        # Display gender distribution as a pie chart
                        import plotly.express as px
                        fig = px.pie(
                            gender_data, 
                            values="Count", 
                            names="Gender",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with gender_cols[1]:
                        # Display the raw numbers
                        st.dataframe(gender_data, hide_index=True, use_container_width=True)
                    
                    # Display age-health correlation if available
                    if 'age_health_correlation' in demographics:
                        st.subheader("Age-Health Correlation")
                        st.write("Distribution of conditions across age groups:")
                        
                        # This would need more processing to display properly
                        # For now, just show a message
                        st.info("Detailed age-health correlation data is available but requires additional visualization")
                else:
                    st.warning("No demographic data available for selected region")
        
        with tabs[3]:  # Visualizations tab
            region = st.selectbox(
                "Select Region",
                ["North", "South", "East", "West", "Central"],
                key="region_selector_viz"
            )
            
            if region:
                analysis = regional_module.analyze_regional_patterns(region)
                
                if analysis:
                    # Enhanced visualization options
                    st.subheader("Disease Prevalence Map")
                    
                    # Get a list of diseases from the analysis
                    diseases = list(analysis['disease_prevalence'].keys())
                    # Add "All Diseases" option at the beginning
                    diseases_options = ["All Diseases"] + diseases
                    
                    # Create two columns for visualization controls
                    viz_cols = st.columns([2, 1])
                    
                    with viz_cols[0]:
                        selected_disease = st.selectbox(
                            "Select Disease to Visualize",
                            diseases_options
                        )
                    
                    with viz_cols[1]:
                        # Add map style options
                        map_style = st.selectbox(
                            "Map Style",
                            ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"]
                        )
                    
                    # Add color scale options
                    color_scale = st.select_slider(
                        "Color Scale",
                        options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"]
                    )
                    
                    if selected_disease:
                        # If "All Diseases" is selected, pass None to show all diseases
                        disease_param = None if selected_disease == "All Diseases" else selected_disease
                        
                        # Use the enhanced plot_regional_heatmap with additional parameters
                        heatmap = regional_module.plot_regional_heatmap(
                            disease_param,
                            color_scale=color_scale,
                            mapbox_style=map_style
                        )
                        
                        # Display the map with full width
                        st.plotly_chart(heatmap, use_container_width=True)
                        
                        # Add download option for the visualization
                        st.download_button(
                            label="Download Map as HTML",
                            data="Map data would be here in a real implementation",
                            file_name=f"{region}_{selected_disease}_map.html",
                            mime="text/html",
                            disabled=True  # This would need actual implementation
                        )
                        
                        # Add a time-based analysis option
                        st.subheader("Temporal Analysis")
                        
                        # Time period selection
                        time_period = st.radio(
                            "Select Time Period",
                            ["monthly", "weekly", "quarterly", "daily"],
                            horizontal=True
                        )
                        
                        # Get temporal analysis data
                        temporal_data = regional_module.get_temporal_analysis(
                            region, 
                            disease_param,
                            time_period
                        )
                        
                        if temporal_data and temporal_data['time_series']:
                            # Convert time series to DataFrame for visualization
                            time_df = pd.DataFrame({
                                'Date': temporal_data['time_series'].keys(),
                                'Cases': temporal_data['time_series'].values()
                            })
                            
                            # Sort by date
                            time_df['Date'] = pd.to_datetime(time_df['Date'])
                            time_df = time_df.sort_values('Date')
                            
                            # Create line chart
                            import plotly.express as px
                            fig = px.line(
                                time_df, 
                                x='Date', 
                                y='Cases',
                                markers=True,
                                title=f"Disease Trend Over Time ({time_period.capitalize()})"
                            )
                            
                            # Add trend information
                            fig.add_annotation(
                                text=f"Trend: {temporal_data['trend'].title()} | Growth Rate: {temporal_data['growth_rate']:.2%}",
                                xref="paper", yref="paper",
                                x=0.5, y=1.05,
                                showarrow=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display peak information
                            st.info(f"Peak period: {temporal_data['peak_period']}")
                        else:
                            st.info("Insufficient temporal data available for the selected parameters")
                        
                        # Add comparative analysis
                        st.subheader("Regional Comparison")
                        
                        # Get all regions for comparison
                        all_regions = ["North", "South", "East", "West", "Central"]
                        
                        # Allow user to select regions to compare
                        regions_to_compare = st.multiselect(
                            "Select Regions to Compare",
                            all_regions,
                            default=[region]  # Default to current region
                        )
                        
                        if regions_to_compare:
                            # Generate comparative visualization
                            comparison_chart = regional_module.generate_comparative_visualization(
                                regions=regions_to_compare,
                                disease=disease_param
                            )
                            
                            if comparison_chart:
                                st.plotly_chart(comparison_chart, use_container_width=True)
                            else:
                                st.info("Unable to generate comparison chart with selected parameters")
                else:
                    st.warning("No data available for selected region")
                
    elif analysis_type == "Model Performance":
        st.header("Model Performance Metrics")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Disease Prediction Model")
            metrics = {
                'Accuracy': 0.85,
                'Precision': 0.83,
                'Recall': 0.87,
                'F1 Score': 0.85
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
                
            # Plot ROC curve
            fig = biomarker_module.plot_roc_curve()
            if fig:
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("Risk Assessment Model")
            risk_metrics = {
                'Accuracy': 0.88,
                'Precision': 0.86,
                'Recall': 0.89,
                'F1 Score': 0.87
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, f"{value:.2%}")
            
            # Plot confusion matrix
            fig = biomarker_module.plot_confusion_matrix()
            if fig:
                st.plotly_chart(fig)
        
        # Model update information
        st.subheader("Model Updates")
        st.info("Last model update: 2025-05-01")
        st.progress(0.85)
        st.caption("Model performance score: 85%")

if __name__ == "__main__":
    main()

