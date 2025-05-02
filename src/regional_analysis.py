import pandas as pd
import plotly.express as px
from typing import Dict, List
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim

class RegionalAnalysis:
    """
    A class for analyzing and visualizing regional health patterns and trends.
    
    This class provides functionality to:
    - Load and analyze regional health data
    - Calculate disease prevalence in regions
    - Analyze demographic patterns
    - Identify disease hotspots
    - Generate visualizations and summary reports
    
    Attributes:
        regional_data (pd.DataFrame): The loaded regional health data
        geolocator (Nominatim): Geocoding service for location operations
    """

    def __init__(self):
        """Initialize a new RegionalAnalysis instance."""
        self.regional_data = None
        self.geolocator = Nominatim(user_agent="health_analysis")
        
    def load_data(self, data_path: str) -> None:
        """
        Load regional health data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file containing regional health data
        """
        self.regional_data = pd.read_csv(data_path)
        
    def analyze_regional_patterns(self, region: str) -> Dict:
        """
        Analyze health patterns in a specific region.
        
        Args:
            region (str): Name of the region to analyze (e.g., 'North', 'South')
            
        Returns:
            Dict: Analysis results containing:
                - disease_prevalence: Disease counts and percentages
                - health_indicators: Statistical analysis of health metrics
                - demographic_patterns: Age and gender distribution analysis
                
        Example:
            {
                'disease_prevalence': {
                    'Diabetes': {'count': 100, 'percentage': 25.0},
                    'Hypertension': {'count': 150, 'percentage': 37.5}
                },
                'health_indicators': {
                    'bmi': {'mean': 24.5, 'median': 23.8, 'std': 4.2}
                },
                'demographic_patterns': {
                    'age_groups': {'18-29': 50, '30-44': 100},
                    'gender_distribution': {'M': 120, 'F': 180}
                }
            }
        """
        if self.regional_data is None:
            return {}
            
        region_data = self.regional_data[
            self.regional_data['region'] == region
        ]
        
        analysis = {
            'disease_prevalence': self._calculate_disease_prevalence(region_data),
            'health_indicators': self._analyze_health_indicators(region_data),
            'demographic_patterns': self._analyze_demographics(region_data)
        }
        
        return analysis
    
    def _calculate_disease_prevalence(self, region_data: pd.DataFrame) -> Dict:
        """
        Calculate disease prevalence in the region.
        
        Args:
            region_data (pd.DataFrame): DataFrame containing regional health data
            
        Returns:
            Dict: Dictionary of diseases with their counts and percentages
                {
                    'disease_name': {
                        'count': number_of_cases,
                        'percentage': percentage_of_total
                    }
                }
        """
        total_patients = len(region_data)
        disease_counts = region_data['diagnosis'].value_counts()
        
        prevalence = {
            disease: {
                'count': count,
                'percentage': (count / total_patients) * 100
            }
            for disease, count in disease_counts.items()
        }
        
        return prevalence
    
    def _analyze_health_indicators(self, region_data: pd.DataFrame) -> Dict:
        """
        Analyze key health indicators in the region.
        
        Args:
            region_data (pd.DataFrame): DataFrame containing regional health data
            
        Returns:
            Dict: Statistical analysis of numeric health indicators
                {
                    'indicator_name': {
                        'mean': mean_value,
                        'median': median_value,
                        'std': standard_deviation
                    }
                }
        """
        indicators = {}
        
        numeric_columns = region_data.select_dtypes(include=[np.number]).columns
        for indicator in numeric_columns:
            if indicator not in ['patient_id', 'latitude', 'longitude']:
                indicators[indicator] = {
                    'mean': region_data[indicator].mean(),
                    'median': region_data[indicator].median(),
                    'std': region_data[indicator].std()
                }
        
        return indicators
    
    def _analyze_demographics(self, region_data: pd.DataFrame) -> Dict:
        """
        Analyze demographic patterns in health outcomes.
        
        Args:
            region_data (pd.DataFrame): DataFrame containing regional health data
            
        Returns:
            Dict: Demographic analysis containing:
                - age_groups: Distribution of patients by age group
                - gender_distribution: Distribution of patients by gender
                - age_health_correlation: Disease prevalence by age group
        """
        demographics = {
            'age_groups': region_data['age_group'].value_counts().to_dict(),
            'gender_distribution': region_data['gender'].value_counts().to_dict()
        }
        
        # Add correlation between demographics and health indicators
        if 'age' in region_data.columns and 'diagnosis' in region_data.columns:
            demographics['age_health_correlation'] = region_data.groupby('age_group')['diagnosis'].value_counts().to_dict()
            
        return demographics
    
    def identify_hotspots(self, condition: str) -> List[Dict]:
        """
        Identify geographical hotspots for specific health conditions using DBSCAN clustering.
        
        Args:
            condition (str): The health condition to analyze
            
        Returns:
            List[Dict]: List of identified hotspots, each containing:
                - center: [latitude, longitude] of cluster center
                - count: Number of cases in the cluster
                - severity: 'low', 'medium', or 'high'
                - recent_cases: Number of cases in last 14 days
                - radius: Radius of the cluster in coordinate units
                
        Notes:
            - Uses DBSCAN clustering with eps=0.1 and min_samples=5
            - Severity levels:
                - low: < 10 cases
                - medium: 10-20 cases
                - high: > 20 cases
        """
        if self.regional_data is None:
            return []
            
        condition_data = self.regional_data[
            self.regional_data['diagnosis'] == condition
        ]
        
        # Use DBSCAN for clustering
        if len(condition_data) > 0 and 'latitude' in condition_data.columns:
            coords = condition_data[['latitude', 'longitude']].values
            clustering = DBSCAN(eps=0.1, min_samples=5).fit(coords)
            
            hotspots = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Exclude noise points
                    cluster_points = coords[clustering.labels_ == cluster_id]
                    cluster_size = len(cluster_points)
                    
                    # Calculate severity based on cluster size and rate of increase
                    severity = 'low'
                    if cluster_size > 10:
                        severity = 'medium'
                    if cluster_size > 20:
                        severity = 'high'
                    
                    # Get recent cases (last 14 days) to check for rapid increase
                    recent_cases = condition_data[
                        pd.to_datetime(condition_data['visit_date']) > 
                        (pd.Timestamp.now() - pd.Timedelta(days=14))
                    ]
                    recent_count = len(recent_cases)
                    
                    hotspots.append({
                        'center': cluster_points.mean(axis=0).tolist(),
                        'count': cluster_size,
                        'severity': severity,
                        'recent_cases': recent_count,
                        'radius': np.max(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1))
                    })
                    
            return hotspots
        return []
    
    def plot_regional_heatmap(self, indicator: str):
        """
        Create a heatmap visualization for health indicators using Plotly.
        
        Args:
            indicator (str): The health indicator to visualize
            
        Returns:
            plotly.graph_objects.Figure or None: Interactive mapbox scatter plot
            showing the distribution of the health indicator across regions.
            Returns None if data is missing or an error occurs.
            
        Notes:
            - Uses OpenStreetMap style for base map
            - Color scale: Viridis
            - Uniform point size: 10
            - Default zoom level: 7
        """
        if self.regional_data is None or indicator not in self.regional_data.columns:
            return None
            
        try:
            # Create a scatter mapbox for better visualization
            fig = px.scatter_mapbox(
                self.regional_data,
                lat='latitude',
                lon='longitude',
                color=indicator,
                size=[10] * len(self.regional_data),  # Uniform size for all points
                color_continuous_scale='Viridis',
                hover_data=['region', indicator],
                zoom=7,
                title=f'{indicator} Distribution by Region',
                mapbox_style='open-street-map'  # Using OpenStreetMap style which doesn't require a token
            )
            
            # Update layout for better visibility
            fig.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                height=600
            )
            
            return fig
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            return None
    
    def generate_region_summary(self, region: str) -> str:
        """
        Generate a text summary report for a specific region.
        
        Args:
            region (str): Name of the region to summarize
            
        Returns:
            str: Formatted summary report containing:
                - Top 5 health conditions with percentages
                - Key health indicators with means and standard deviations
                
        Example:
            Health Analysis Summary for North

            Top 5 Health Conditions:
            - Hypertension: 25.5%
            - Diabetes: 18.2%
            - Obesity: 15.7%
            - Heart Disease: 12.3%
            - Asthma: 8.8%

            Key Health Indicators:
            - BMI: 26.4 (±4.8)
            - Blood Pressure: 128.5 (±15.2)
        """
        if self.regional_data is None:
            return "No data available"
            
        analysis = self.analyze_regional_patterns(region)
        
        summary = f"Health Analysis Summary for {region}\n\n"
        
        # Add top diseases
        top_diseases = sorted(
            analysis['disease_prevalence'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
        
        summary += "Top 5 Health Conditions:\n"
        for disease, stats in top_diseases:
            summary += f"- {disease}: {stats['percentage']:.1f}%\n"
            
        # Add key health indicators
        if analysis['health_indicators']:
            summary += "\nKey Health Indicators:\n"
            for indicator, stats in analysis['health_indicators'].items():
                summary += f"- {indicator}: {stats['mean']:.2f} (±{stats['std']:.2f})\n"
                
        return summary