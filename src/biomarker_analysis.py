import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class BiomarkerAnalysis:
    def __init__(self):
        self.biomarker_data = None
        self.reference_ranges = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str, reference_path: str) -> None:
        """Load biomarker data and reference ranges"""
        self.biomarker_data = pd.read_csv(data_path)
        self.reference_ranges = pd.read_csv(reference_path)
    
    def analyze_biomarkers(self, patient_id: str) -> Dict:
        """Analyze biomarker trends and identify abnormal values"""
        if self.biomarker_data is None:
            return {}
            
        patient_markers = self.biomarker_data[
            self.biomarker_data['patient_id'] == patient_id
        ]
        
        analysis_results = {
            'abnormal_markers': self._identify_abnormal_markers(patient_markers),
            'trend_analysis': self._analyze_trends(patient_markers),
            'risk_factors': self._assess_risk_factors(patient_markers)
        }
        
        return analysis_results
    
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
    
    def train_and_evaluate_model(self, patient_data: pd.DataFrame = None, test_size: float = 0.2, n_estimators: int = 100) -> Dict:
        """
        Train a model and evaluate its performance on disease prediction
        
        Args:
            patient_data: Optional DataFrame to use instead of stored data
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            n_estimators: Number of trees in the Random Forest
            
        Returns:
            Dict containing performance metrics and model insights
        """
        if self.biomarker_data is None:
            return {}
            
        # Load and merge patient data to get diagnoses
        patient_data = pd.read_csv('data/patient_data.csv')
        data = pd.merge(
            self.biomarker_data,
            patient_data[['patient_id', 'visit_date', 'diagnosis']],
            on=['patient_id'],
            how='inner'
        )
        
        # Ensure dates match between biomarker and patient data
        data['date'] = pd.to_datetime(data['date'])
        data['visit_date'] = pd.to_datetime(data['visit_date'])
        data = data[data['date'] == data['visit_date']]
        
        if len(data) == 0:
            return {}
        
        # Prepare features
        features = ['cholesterol', 'blood_pressure', 'glucose', 'hemoglobin', 'white_blood_cells']
        X = data[features]
        y = data['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'f1_scores': {
                'macro': f1_score(y_test, y_pred, average='macro'),
                'weighted': f1_score(y_test, y_pred, average='weighted')
            },
            'recall_scores': {
                'macro': recall_score(y_test, y_pred, average='macro'),
                'weighted': recall_score(y_test, y_pred, average='weighted')
            },
            'accuracy': accuracy_score(y_test, y_pred),
            'feature_importance': dict(zip(features, self.model.feature_importances_))
        }
        
        return metrics
    
    def plot_confusion_matrix(self, metrics: Dict):
        """Plot confusion matrix using seaborn"""
        if 'confusion_matrix' not in metrics:
            return None
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix for Disease Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()