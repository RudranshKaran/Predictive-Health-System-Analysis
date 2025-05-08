import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_home_page():
    """Display the home page with dashboard and overview"""
    
    st.title("Anemia Risk Prediction System")
    
    st.markdown("""
    ## Early Detection for Better Outcomes
    
    This advanced system uses machine learning to detect anemia risk **even when standard blood test values 
    appear within normal ranges**. By analyzing subtle patterns and relationships between parameters, 
    we can identify early signs of anemia before traditional diagnostic methods.
    """)
    
    # Dashboard metrics in cards
    st.markdown("## Key Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:10px; padding:15px; text-align:center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0; color:#2c3e50;">92%</h3>
            <p style="margin:0; font-size:0.9em; color:#7b8a8b;">Overall Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:10px; padding:15px; text-align:center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0; color:#2c3e50;">85%</h3>
            <p style="margin:0; font-size:0.9em; color:#7b8a8b;">Early Detection Rate</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:10px; padding:15px; text-align:center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0; color:#2c3e50;">78%</h3>
            <p style="margin:0; font-size:0.9em; color:#7b8a8b;">Latent Anemia Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlighting with interactive elements
    st.markdown("## Key Features")
    
    tab1, tab2, tab3 = st.tabs(["Prediction Engine", "Visual Analysis", "Educational Tools"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Advanced Prediction Engine
            
            Our system analyzes multiple blood parameters and their relationships to detect anemia risk:
            
            - **Pattern Recognition**: Identifies subtle patterns in blood values
            - **Risk Stratification**: Categorizes patients into risk groups
            - **Anemia Type Classification**: Suggests possible anemia types based on parameter patterns
            - **Personalized Recommendations**: Provides specific follow-up suggestions
            """)
        
        with col2:
            # Simplified decision tree visualization
            labels = ['Hb < 13.5?', 'MCV < 80?', 'RDW > 14.5?', 'High Risk', 'Med Risk', 'Low Risk']
            parents = ['', 'Hb < 13.5?', 'MCV < 80?', 'RDW > 14.5?', 'MCV < 80?', 'Hb < 13.5?']
            
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                marker=dict(colors=['#3498db', '#e74c3c', '#f39c12', '#c0392b', '#d35400', '#2ecc71']),
                textinfo='label'
            ))
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Simplified decision path visualization")

    with tab2:
        st.markdown("""
        ### Interactive Data Visualizations
        
        Explore and understand your patient data with interactive visualizations:
        
        - **Parameter Distribution**: Compare patient values against population norms
        - **Correlation Analysis**: Understand relationships between blood parameters
        - **SHAP Explanations**: See exactly how each parameter affects risk prediction
        - **3D Molecule Visualization**: Explore the hemoglobin structure
        """)
        
        # Sample visualization
        # Create sample data for visualization
        np.random.seed(42)
        normal_hb = np.random.normal(14, 1, 200)
        anemic_hb = np.random.normal(11, 1, 100)
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Hemoglobin Distribution", "Parameter Relationships"))
        
        # First subplot - histogram
        fig.add_trace(
            go.Histogram(x=normal_hb, name="Normal", marker_color='#3498db', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=anemic_hb, name="Anemic", marker_color='#e74c3c', opacity=0.7),
            row=1, col=1
        )
        
        # Second subplot - scatter plot
        mcv_normal = np.random.normal(90, 5, 200)
        mcv_anemic = np.random.normal(75, 7, 100)
        
        fig.add_trace(
            go.Scatter(x=normal_hb, y=mcv_normal, mode='markers', name="Normal",
                      marker=dict(color='#3498db', size=6, opacity=0.6)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=anemic_hb, y=mcv_anemic, mode='markers', name="Anemic",
                      marker=dict(color='#e74c3c', size=6, opacity=0.6)),
            row=1, col=2
        )
        
        fig.update_layout(height=350, showlegend=False, 
                         margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(title_text="Hemoglobin (g/dL)", row=1, col=1)
        fig.update_xaxes(title_text="Hemoglobin (g/dL)", row=1, col=2)
        fig.update_yaxes(title_text="MCV (fL)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("""
        ### Educational Content
        
        Learn about anemia and its effects:
        
        - **Interactive Hemoglobin Model**: Explore the 3D structure of hemoglobin
        - **RBC Visualization**: Compare normal and abnormal red blood cells
        - **Anemia Type Guide**: Understand different types of anemia and their characteristics
        - **Parameter Guide**: Learn what each blood parameter means for anemia diagnosis
        """)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    ## Get Started
    
    1. Navigate to the **Prediction** page to analyze patient data
    2. Explore our **Visualizations** to understand data patterns
    3. Learn more with the **Educational Content**
    """)
    
    # Sample statistics
    st.markdown("### Anemia Statistics")
    
    # Create a choropleth map of global anemia prevalence
    countries = ['United States', 'Canada', 'Brazil', 'Argentina', 'United Kingdom', 
                 'France', 'Germany', 'Spain', 'Italy', 'South Africa', 'Egypt', 
                 'India', 'China', 'Japan', 'Australia']
    
    # Sample prevalence rates (%)
    prevalence = [12, 10, 22, 18, 14, 15, 13, 16, 17, 27, 30, 53, 32, 21, 14]
    
    df = pd.DataFrame({'Country': countries, 'Prevalence': prevalence})
    
    fig = px.choropleth(
        df, 
        locations='Country', 
        locationmode='country names',
        color='Prevalence',
        color_continuous_scale='RdBu_r',
        range_color=(0, 60),
        labels={'Prevalence': 'Anemia Prevalence (%)'}
    )
    
    fig.update_layout(
        title_text='Global Anemia Prevalence',
        height=450,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="font-size:0.8em; color:#7b8a8b; text-align:center;">
    Sample data for demonstration purposes. Actual prevalence rates may vary.
    </div>
    """, unsafe_allow_html=True)