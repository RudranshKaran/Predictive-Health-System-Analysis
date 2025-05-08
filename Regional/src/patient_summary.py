import pandas as pd
import plotly.express as px
from typing import Dict, List
from datetime import datetime

class PatientSummary:
    def __init__(self):
        self.patient_data = None
        
    def load_data(self, data_path: str) -> None:
        """Load patient data from CSV file"""
        self.patient_data = pd.read_csv(data_path)
        # Convert date strings to datetime objects
        self.patient_data['visit_date'] = pd.to_datetime(self.patient_data['visit_date'])
        
    def get_visit_timeline(self, patient_id: str) -> List[Dict]:
        """Get a timeline of patient visits"""
        if self.patient_data is None:
            return []
            
        patient_visits = self.patient_data[
            self.patient_data['patient_id'] == patient_id
        ].sort_values('visit_date')
        
        timeline = []
        for _, visit in patient_visits.iterrows():
            timeline.append({
                'date': visit['visit_date'].strftime('%Y-%m-%d'),
                'diagnosis': visit['diagnosis'],
                'medications': visit['medications']
            })
            
        return timeline
    
    def get_recurring_illnesses(self, patient_id: str) -> List[str]:
        """Identify recurring illnesses for a patient"""
        if self.patient_data is None:
            return []
            
        patient_diagnoses = self.patient_data[
            self.patient_data['patient_id'] == patient_id
        ]['diagnosis'].value_counts()
        
        # Consider an illness recurring if it appears more than once
        recurring = patient_diagnoses[patient_diagnoses > 1].index.tolist()
        return recurring
    
    def plot_visit_history(self, patient_id: str):
        """Create an interactive timeline visualization of patient visits"""
        if self.patient_data is None:
            return None
            
        patient_visits = self.patient_data[
            self.patient_data['patient_id'] == patient_id
        ].sort_values('visit_date')
        
        if len(patient_visits) == 0:
            return None
            
        # Create a more informative timeline plot
        fig = px.scatter(
            patient_visits,
            x='visit_date',
            y='diagnosis',
            text='medications',
            title=f'Visit History for Patient {patient_id}',
            height=400
        )
        
        # Customize the layout
        fig.update_traces(
            marker=dict(size=12, symbol='diamond'),
            textposition='top center'
        )
        
        fig.update_layout(
            xaxis_title='Visit Date',
            yaxis_title='Diagnosis',
            showlegend=False,
            yaxis={'categoryorder': 'category ascending'},
            hovermode='x'
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Diagnosis: %{y}",
                "Medications: %{text}",
                "<extra></extra>"
            ])
        )
        
        return fig
    
    def get_patient_demographics(self, patient_id: str) -> Dict:
        """Get demographic information for a patient"""
        if self.patient_data is None:
            return {}
            
        patient_info = self.patient_data[
            self.patient_data['patient_id'] == patient_id
        ].iloc[0]
        
        return {
            'age': patient_info['age'],
            'gender': patient_info['gender']
        }
    
    def analyze_medication_history(self, patient_id: str) -> Dict:
        """Analyze patient's medication history"""
        if self.patient_data is None:
            return {}
            
        patient_meds = self.patient_data[
            self.patient_data['patient_id'] == patient_id
        ]['medications'].value_counts()
        
        return {
            'current_medications': patient_meds.index.tolist(),
            'medication_frequency': patient_meds.to_dict()
        }
    
    def generate_summary_report(self, patient_id: str) -> str:
        """Generate a comprehensive summary report for a patient"""
        if self.patient_data is None:
            return "No data available"
            
        demographics = self.get_patient_demographics(patient_id)
        timeline = self.get_visit_timeline(patient_id)
        medications = self.analyze_medication_history(patient_id)
        recurring = self.get_recurring_illnesses(patient_id)
        
        report = f"Patient Summary Report - ID: {patient_id}\n\n"
        report += f"Demographics:\n"
        report += f"- Age: {demographics['age']}\n"
        report += f"- Gender: {demographics['gender']}\n\n"
        
        report += "Visit History:\n"
        for visit in timeline:
            report += f"- {visit['date']}: {visit['diagnosis']}\n"
            
        if recurring:
            report += "\nRecurring Conditions:\n"
            for condition in recurring:
                report += f"- {condition}\n"
                
        report += "\nCurrent Medications:\n"
        for med in medications['current_medications']:
            freq = medications['medication_frequency'][med]
            report += f"- {med} (prescribed {freq} times)\n"
            
        return report