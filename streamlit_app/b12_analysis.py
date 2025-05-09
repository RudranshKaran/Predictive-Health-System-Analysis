import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def show_b12_analysis():
    """Specialized analysis for B12/folate deficiency anemia"""
    
    st.title("B12/Folate Deficiency Analysis")
    
    st.markdown("""
    ## Specialized B12/Folate Deficiency Detection
    
    Vitamin B12 (cobalamin) and folate deficiencies can lead to macrocytic anemia with distinct 
    characteristics in blood parameters. Our model can detect subtle patterns that indicate 
    early B12/folate deficiencies even before clinical symptoms appear.
    """)
    
    # Visual comparison of normal vs. B12 deficient blood profiles
    st.subheader("Blood Parameter Profile: Normal vs. B12 Deficient")
    
    # Sample data for radar chart
    categories = ['Hemoglobin', 'MCV', 'MCH', 'MCHC', 'RDW', 'Homocysteine']
    
    # Values normalized to 0-1 scale where 0.5 is the middle of normal range
    normal_values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    b12_deficient = [0.35, 0.8, 0.6, 0.4, 0.7, 0.9]  # Higher MCV, higher RDW
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        name='Normal Profile',
        line_color='#3498db'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=b12_deficient,
        theta=categories,
        fill='toself',
        name='B12 Deficient',
        line_color='#e74c3c'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    The radar chart above compares the typical pattern of blood parameters in normal vs. B12 deficient individuals.
    Note the distinctive pattern with elevated MCV (macrocytosis) and RDW in B12 deficiency.
    """)
    
    # B12 pathway visualization
    st.subheader("B12 Metabolism Pathway")
    
    # Create a simple directed graph of B12 metabolism
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Key Role in DNA Synthesis
        
        B12 deficiency affects DNA synthesis and leads to:
        
        - **Impaired cell division**: Resulting in larger but fewer red blood cells
        - **Nuclear maturation defects**: Leading to hypersegmented neutrophils
        - **Ineffective erythropoiesis**: Causing premature destruction of RBC precursors
        - **Elevated homocysteine**: Due to reduced conversion to methionine
        - **Increased methylmalonic acid**: From impaired methylmalonyl-CoA conversion
        
        These metabolic disruptions create the characteristic pattern of macrocytic anemia
        with specific ratios between MCV, MCH and RDW that our model is trained to detect.
        """)
    
    with col2:
        # Create a simplified B12 pathway diagram
        nodes = ['Dietary B12', 'Intrinsic Factor', 'Ileum Absorption', 'Transport (TC-II)', 
                'Cellular Uptake', 'DNA Synthesis']
        edges = [(0,1), (1,2), (2,3), (3,4), (4,5)]
        
        # Create positions for nodes
        pos = {
            0: [0, 5],
            1: [0, 4],
            2: [0, 3],
            3: [0, 2],
            4: [0, 1],
            5: [0, 0]
        }
        
        # Create figure
        fig = go.Figure()
        
        # Add edges as lines
        for edge in edges:
            fig.add_trace(
                go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0]],
                    y=[pos[edge[0]][1], pos[edge[1]][1]],
                    mode='lines',
                    line=dict(width=2, color='#3498db'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Add nodes as markers
        x_nodes = [pos[i][0] for i in range(len(nodes))]
        y_nodes = [pos[i][1] for i in range(len(nodes))]
        
        fig.add_trace(
            go.Scatter(
                x=x_nodes,
                y=y_nodes,
                mode='markers+text',
                marker=dict(size=25, color='#3498db'),
                text=nodes,
                textposition="middle right",
                hoverinfo='text',
                showlegend=False
            )
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=10),
            height=400
        )
        
        st.plotly_chart(fig)
    
    # Risk factors and detection
    st.markdown("---")
    st.subheader("Risk Factors and Early Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Common Risk Factors
        
        - **Age over 60**: Decreased absorption
        - **Vegetarian/vegan diet**: Limited dietary intake
        - **Pernicious anemia**: Autoimmune condition affecting intrinsic factor
        - **Gastric surgery**: Reduced intrinsic factor production
        - **Intestinal disorders**: Malabsorption (Crohn's, celiac disease)
        - **Medications**: PPIs, metformin, etc.
        - **Genetic factors**: MTHFR variants
        """)
        
    with col2:
        st.markdown("""
        ### Early Detection Markers
        
        Our model looks for these early signs:
        
        - **Elevated MCV**: Even at the upper end of normal range
        - **Rising RDW**: Indicating variable cell sizes
        - **MCH/MCHC patterns**: Subtle changes in these ratios
        - **Normal hemoglobin**: B12 deficiency can exist before anemia develops
        - **Age and gender interactions**: Different thresholds based on demographics
        """)
    
    # Interactive case studies
    st.markdown("---")
    st.subheader("Interactive Case Studies")
    
    # Create tabs for different case studies
    case1, case2 = st.tabs(["Early B12 Deficiency", "Advanced B12 Deficiency"])
    
    with case1:
        st.markdown("""
        ### Case Study: Early B12 Deficiency
        
        **Patient Profile:**
        - 58-year-old female
        - Vegetarian diet for 12 years
        - Mild fatigue and occasional tingling in feet
        - No visible anemia on standard blood tests
        """)
        
        # Create a table of values
        early_case = pd.DataFrame({
            'Parameter': ['Hemoglobin', 'MCV', 'MCH', 'MCHC', 'RDW', 'B12 Level', 'Homocysteine', 'Risk Score'],
            'Value': ['12.2 g/dL', '98 fL', '31 pg', '33 g/dL', '15.1%', '220 pg/mL', '14.2 μmol/L', '0.68'],
            'Status': ['Low Normal', 'High Normal', 'Normal', 'Normal', 'High', 'Low Normal', 'High', 'Moderate Risk'],
            'Reference': ['12.0-15.5', '80-100', '27-33', '32-36', '11.5-14.5', '200-900', '<12', '<0.5']
        })
        
        st.dataframe(early_case.style.apply(lambda x: ['background-color: #ffcccc' if x == 'High' or x == 'Low' 
                                                    else 'background-color: #ffffcc' if 'Normal' in x and ('High' in x or 'Low' in x) 
                                                    else 'background-color: #ccffcc' if x == 'Normal'
                                                    else '' for x in early_case['Status']], axis=0), use_container_width=True)
        
        st.markdown("""
        **Key Findings:**
        
        This patient demonstrates early B12 deficiency despite having "normal" values in standard tests. 
        Our model detected concerning patterns: MCV at the upper end of normal, elevated RDW, 
        and the combination of low-normal B12 with elevated homocysteine.
        
        **Model Detection:**
        
        The algorithm recognized this pattern as characteristic of early B12 deficiency, 
        assigning a risk score of 0.68 (moderate risk). This patient would benefit from 
        B12 supplementation before developing clinical anemia.
        """)
        
    with case2:
        st.markdown("""
        ### Case Study: Advanced B12 Deficiency
        
        **Patient Profile:**
        - 72-year-old male
        - History of partial gastrectomy 15 years ago
        - Fatigue, weakness, and difficulty walking
        - Glossitis and cognitive changes
        """)
        
        # Create a table of values
        advanced_case = pd.DataFrame({
            'Parameter': ['Hemoglobin', 'MCV', 'MCH', 'MCHC', 'RDW', 'B12 Level', 'Homocysteine', 'Risk Score'],
            'Value': ['10.2 g/dL', '112 fL', '35 pg', '31 g/dL', '17.8%', '120 pg/mL', '22.5 μmol/L', '0.95'],
            'Status': ['Low', 'High', 'High', 'Low', 'High', 'Low', 'High', 'High Risk'],
            'Reference': ['13.5-17.5', '80-100', '27-33', '32-36', '11.5-14.5', '200-900', '<12', '<0.5']
        })
        
        st.dataframe(advanced_case.style.apply(lambda x: ['background-color: #ffcccc' if x == 'High' or x == 'Low' 
                                                       else 'background-color: #ffffcc' if 'Normal' in x and ('High' in x or 'Low' in x) 
                                                       else 'background-color: #ccffcc' if x == 'Normal'
                                                       else '' for x in advanced_case['Status']], axis=0), use_container_width=True)
        
        st.markdown("""
        **Key Findings:**
        
        This patient shows classic advanced B12 deficiency anemia with frank macrocytosis (high MCV), 
        low hemoglobin, and dramatically elevated homocysteine. The peripheral blood smear would likely show 
        hypersegmented neutrophils and macroovalocytes.
        
        **Model Detection:**
        
        The algorithm assigned a high risk score of 0.95, recognizing the characteristic pattern 
        of B12 deficiency anemia. Urgent B12 replacement therapy is indicated, with monitoring for 
        neurological improvement.
        """)
    
    # Educational information on testing
    st.markdown("---")
    st.subheader("Recommended Testing")
    
    st.info("""
    For suspected B12 deficiency, we recommend:
    
    1. **Serum B12 levels**: May be falsely normal in some cases
    2. **Methylmalonic acid (MMA)**: More sensitive marker for functional B12 deficiency
    3. **Homocysteine levels**: Elevated in both B12 and folate deficiency
    4. **Complete blood count with peripheral smear**: To examine cell morphology
    5. **Intrinsic factor antibodies**: To check for pernicious anemia
    """)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    ## Next Steps
    
    If your patient shows signs of potential B12 deficiency:
    
    1. Use our **Prediction** tool to analyze their complete blood parameters
    2. Review the **SHAP analysis** to understand which factors are driving the risk
    3. Consider specialized B12 and folate testing for confirmation
    4. Implement early intervention to prevent neurological complications
    """)