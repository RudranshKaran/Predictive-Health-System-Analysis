import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.express as px
from sklearn.preprocessing import StandardScaler

class BiomarkerAnalysis:
    def __init__(self):
        self.biomarker_data = None
        self.reference_ranges = None
        
    def load_data(self, data_path: str, reference_path: str) -> None:
        """Load biomarker data and reference ranges"""
        try:
            self.biomarker_data = pd.read_csv(data_path)
            self.reference_ranges = pd.read_csv(reference_path)
            
            # Validate required columns
            required_biomarker_columns = ['patient_id', 'date']
            required_reference_columns = ['biomarker', 'min_value', 'max_value']
            
            if not all(col in self.biomarker_data.columns for col in required_biomarker_columns):
                raise ValueError(f"Biomarker data missing required columns: {required_biomarker_columns}")
            
            if not all(col in self.reference_ranges.columns for col in required_reference_columns):
                raise ValueError(f"Reference ranges missing required columns: {required_reference_columns}")
            
            # Handle missing values
            self.biomarker_data = self.biomarker_data.dropna(subset=['patient_id'])
            numeric_columns = self.biomarker_data.select_dtypes(include=[np.number]).columns
            self.biomarker_data[numeric_columns] = self.biomarker_data[numeric_columns].fillna(method='ffill')
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find data file: {str(e)}")
        except pd.errors.EmptyDataError:
            raise ValueError("Data file is empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing data file - invalid format")
            
    def analyze_biomarkers(self, patient_id: str) -> Dict:
        """Analyze biomarker trends and identify abnormal values"""
        if self.biomarker_data is None or self.reference_ranges is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        patient_markers = self.biomarker_data[
            self.biomarker_data['patient_id'] == patient_id
        ]
        
        if patient_markers.empty:
            return {}
            
        try:
            analysis_results = {
                'abnormal_markers': self._identify_abnormal_markers(patient_markers),
                'trend_analysis': self._analyze_trends(patient_markers),
                'risk_factors': self._assess_risk_factors(patient_markers)
            }
            return analysis_results
        except Exception as e:
            raise ValueError(f"Error analyzing biomarker data: {str(e)}")
    
    def _identify_abnormal_markers(self, patient_data: pd.DataFrame) -> List[Dict]:
        """Identify biomarkers outside reference ranges"""
        abnormal_markers = []
        
        for marker in self.reference_ranges['biomarker'].unique():
            if marker in patient_data.columns:
                ref_range = self.reference_ranges[
                    self.reference_ranges['biomarker'] == marker
                ].iloc[0]
                
                latest_value = patient_data[marker].iloc[-1]
                
                if latest_value < ref_range['min_value'] or \
                   latest_value > ref_range['max_value']:
                    abnormal_markers.append({
                        'marker': marker,
                        'value': latest_value,
                        'reference_range': f"{ref_range['min_value']}-{ref_range['max_value']}"
                    })
                    
        return abnormal_markers
    
    def _analyze_trends(self, patient_data: pd.DataFrame) -> Dict:
        """Analyze trends in biomarker values over time"""
        trends = {}
        
        for marker in patient_data.select_dtypes(include=[np.number]).columns:
            if marker != 'patient_id':
                values = patient_data[marker].values
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    trends[marker] = {
                        'trend': 'increasing' if trend > 0 else 'decreasing',
                        'magnitude': abs(trend)
                    }
                    
        return trends
    
    def _assess_risk_factors(self, patient_data: pd.DataFrame) -> List[str]:
        """Identify potential health risk factors based on biomarker patterns"""
        risk_factors = []
        latest_values = patient_data.iloc[-1]
        
        # Example risk factor rules (to be expanded based on medical guidelines)
        if 'cholesterol' in latest_values and latest_values['cholesterol'] > 200:
            risk_factors.append('High Cholesterol')
        if 'blood_pressure' in latest_values and latest_values['blood_pressure'] > 140:
            risk_factors.append('High Blood Pressure')
        
        return risk_factors
    
    def plot_biomarker_trends(self, patient_id: str, marker: str):
        """Create an interactive plot for biomarker trends"""
        if self.biomarker_data is None:
            return None
            
        patient_data = self.biomarker_data[
            self.biomarker_data['patient_id'] == patient_id
        ]
        
        fig = px.line(patient_data, 
                     x='date',
                     y=marker,
                     title=f'{marker} Trend Analysis for Patient {patient_id}')
        return fig
    
    def plot_roc_curve(self):
        """Create ROC curve plot for model performance visualization"""
        # Create sample ROC curve using plotly
        fig = px.line(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1],
            y=[0, 0.3, 0.5, 0.7, 0.9, 1],
            title='ROC Curve',
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        return fig
    
    def plot_confusion_matrix(self):
        """Create confusion matrix visualization"""
        # Create sample confusion matrix using plotly
        matrix = [[85, 15], [12, 88]]
        fig = px.imshow(
            matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            text=matrix,
            aspect="auto",
            title="Confusion Matrix"
        )
        return fig