import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from src.gemini_service import GeminiService

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
        try:
            self.gemini_service = GeminiService()
            self.gemini_available = True
        except Exception as e:
            print(f"Warning: Gemini service initialization failed: {str(e)}")
            self.gemini_available = False
        
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
                    
                    # Get the region for this cluster
                    # Find the most common region in the cluster
                    cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
                    cluster_regions = condition_data.iloc[cluster_indices]['region'].value_counts()
                    most_common_region = cluster_regions.index[0] if len(cluster_regions) > 0 else "Unknown"
                    
                    hotspots.append({
                        'center': cluster_points.mean(axis=0).tolist(),
                        'count': cluster_size,
                        'severity': severity,
                        'recent_cases': recent_count,
                        'radius': np.max(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)),
                        'region': most_common_region
                    })
                    
            return hotspots
        return []
        
    def analyze_hotspot_causes(self, condition: str, region: str, hotspot_id: int = 0) -> Dict[str, Any]:
        """
        Analyze the potential root causes of a disease hotspot using Gemini AI.
        
        Args:
            condition (str): The health condition to analyze
            region (str): The region to analyze
            hotspot_id (int): The index of the hotspot to analyze (if multiple exist)
            
        Returns:
            Dict containing:
                - root_causes: List of potential root causes with confidence levels
                - interventions: List of recommended interventions with priority levels
                - urgency_level: Assessed urgency level (low, medium, high, critical)
                - estimated_impact: Estimated impact of interventions
                - error: Error message if analysis failed
        """
        if not self.gemini_available:
            return {
                "error": "Gemini service is not available. Check API key configuration.",
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }
            
        # Get hotspots for the condition
        hotspots = self.identify_hotspots(condition)
        
        if not hotspots or hotspot_id >= len(hotspots):
            return {
                "error": f"No hotspot found for {condition} at index {hotspot_id}",
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }
            
        # Get the specified hotspot
        hotspot = hotspots[hotspot_id]
        
        # Get regional analysis for context
        regional_analysis = self.analyze_regional_patterns(region)
        
        try:
            # Call Gemini service to analyze the hotspot
            analysis = self.gemini_service.analyze_disease_hotspot(
                disease=condition,
                region=region,
                hotspot_data=hotspot,
                regional_indicators=regional_analysis
            )
            
            # Check if the analysis contains an error
            if "error" in analysis:
                print(f"Gemini API error: {analysis['error']}")
                if "details" in analysis:
                    print(f"Details: {analysis['details']}")
                
                # Return the error in a structured format
                return {
                    "error": f"Analysis failed: {analysis.get('error')}",
                    "details": analysis.get("details", "No details available"),
                    "root_causes": [],
                    "interventions": [],
                    "urgency_level": "unknown",
                    "estimated_impact": ""
                }
            
            return analysis
        except Exception as e:
            print(f"Error analyzing hotspot causes: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }
    
    def plot_regional_heatmap(self, disease: str = None, color_scale: str = 'Viridis', mapbox_style: str = 'open-street-map'):
        """
        Create a heatmap visualization for disease prevalence using Plotly.
        
        Args:
            disease (str, optional): The specific disease to visualize.
                If None, visualizes all diseases with color indicating prevalence.
            color_scale (str, optional): Color scale for the visualization.
                Options include 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo'.
            mapbox_style (str, optional): Style of the base map.
                Options include 'open-street-map', 'carto-positron', 'carto-darkmatter', 'stamen-terrain'.
            
        Returns:
            plotly.graph_objects.Figure or None: Interactive mapbox scatter plot
            showing the distribution of diseases across regions.
            Returns None if data is missing or an error occurs.
            
        Notes:
            - Uses customizable map style for base map
            - Customizable color scale
            - Point size indicates case count
            - Default zoom level: 7
            - Improved hover information
        """
        if self.regional_data is None:
            return None
            
        try:
            # If specific disease is provided, filter for that disease
            if disease and disease in self.regional_data['diagnosis'].unique():
                plot_data = self.regional_data[self.regional_data['diagnosis'] == disease]
                # Group by coordinates and count occurrences
                plot_data = plot_data.groupby(['latitude', 'longitude', 'region']).size().reset_index(name='count')
                size_col = 'count'
                color_col = 'count'
                title = f'{disease} Distribution by Region'
                
                # Add percentage of total cases for this disease
                total_cases = plot_data['count'].sum()
                plot_data['percentage'] = (plot_data['count'] / total_cases * 100).round(1)
            else:
                # Create a dataframe of all diseases with counts by location
                plot_data = self.regional_data.groupby(['latitude', 'longitude', 'region', 'diagnosis']).size().reset_index(name='count')
                
                # Add percentage calculation
                total_cases = plot_data['count'].sum()
                plot_data['percentage'] = (plot_data['count'] / total_cases * 100).round(1)
                
                size_col = 'count'
                color_col = 'count'
                title = 'Disease Prevalence by Region'
            
            # Create a scatter mapbox for visualization
            fig = px.scatter_mapbox(
                plot_data,
                lat='latitude',
                lon='longitude',
                color=color_col,
                size=size_col,
                size_max=20,
                color_continuous_scale=color_scale,
                hover_data=['region', 'count', 'percentage'],
                hover_name='region' if 'diagnosis' not in plot_data.columns else 'diagnosis',
                custom_data=['percentage'] if 'percentage' in plot_data.columns else None,
                zoom=7,
                title=title,
                mapbox_style=mapbox_style
            )
            
            # Update hover template to show percentage
            if 'percentage' in plot_data.columns:
                fig.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>Region: %{customdata[0]}%<br>Count: %{marker.size}<br>Percentage: %{customdata[0]}%<extra></extra>'
                )
            
            # Update layout for better visibility
            fig.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0},
                height=600,
                title={
                    'text': title,
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20}
                },
                legend_title_text='Case Count'
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
        
    def get_temporal_analysis(self, region: str, disease: str = None, time_period: str = 'monthly'):
        """
        Analyze disease trends over time for a specific region.
        
        Args:
            region (str): Name of the region to analyze
            disease (str, optional): Specific disease to analyze. If None, analyzes all diseases.
            time_period (str): Aggregation period - 'daily', 'weekly', 'monthly', or 'quarterly'
            
        Returns:
            Dict: Temporal analysis results containing:
                - time_series: Dict mapping time periods to case counts
                - trend: Overall trend description ('increasing', 'decreasing', 'stable')
                - peak_period: Period with highest case count
                - growth_rate: Rate of change over the entire period
        """
        if self.regional_data is None:
            return None
            
        # Filter data for the specified region
        region_data = self.regional_data[self.regional_data['region'] == region].copy()
        
        # Filter for specific disease if provided
        if disease:
            region_data = region_data[region_data['diagnosis'] == disease]
            
        # Ensure visit_date is datetime
        if 'visit_date' in region_data.columns:
            region_data['visit_date'] = pd.to_datetime(region_data['visit_date'])
            
            # Group by time period
            if time_period == 'daily':
                grouped = region_data.groupby(region_data['visit_date'].dt.date).size()
            elif time_period == 'weekly':
                grouped = region_data.groupby(pd.Grouper(key='visit_date', freq='W')).size()
            elif time_period == 'quarterly':
                grouped = region_data.groupby(pd.Grouper(key='visit_date', freq='Q')).size()
            else:  # default to monthly
                grouped = region_data.groupby(pd.Grouper(key='visit_date', freq='M')).size()
                
            # Convert to dictionary with string dates as keys
            time_series = {str(date)[:10]: count for date, count in grouped.items()}
            
            # Calculate trend
            if len(grouped) > 1:
                first_count = grouped.iloc[0]
                last_count = grouped.iloc[-1]
                growth_rate = ((last_count - first_count) / first_count) if first_count > 0 else 0
                
                if growth_rate > 0.1:
                    trend = 'increasing'
                elif growth_rate < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                    
                peak_period = str(grouped.idxmax())[:10]
                
                return {
                    'time_series': time_series,
                    'trend': trend,
                    'peak_period': peak_period,
                    'growth_rate': growth_rate
                }
                
        return None
        
    def generate_comparative_visualization(self, regions=None, disease=None):
        """
        Generate a comparative visualization of disease prevalence across multiple regions.
        
        Args:
            regions (List[str], optional): List of regions to compare. If None, uses all regions.
            disease (str, optional): Specific disease to compare. If None, compares total case counts.
            
        Returns:
            plotly.graph_objects.Figure: Bar chart comparing disease prevalence across regions
        """
        if self.regional_data is None:
            return None
            
        try:
            # If no regions specified, use all unique regions
            if not regions:
                regions = self.regional_data['region'].unique()
                
            # Prepare data for comparison
            comparison_data = []
            
            for region in regions:
                region_data = self.regional_data[self.regional_data['region'] == region]
                
                if disease:
                    # Count cases of specific disease
                    case_count = len(region_data[region_data['diagnosis'] == disease])
                    comparison_data.append({
                        'Region': region,
                        'Cases': case_count,
                        'Disease': disease
                    })
                else:
                    # Group by diagnosis and count
                    diagnoses = region_data['diagnosis'].value_counts()
                    for diagnosis, count in diagnoses.items():
                        comparison_data.append({
                            'Region': region,
                            'Cases': count,
                            'Disease': diagnosis
                        })
            
            # Create DataFrame for visualization
            df = pd.DataFrame(comparison_data)
            
            # Create visualization
            import plotly.express as px
            
            if disease:
                # Simple bar chart for single disease across regions
                fig = px.bar(
                    df,
                    x='Region',
                    y='Cases',
                    title=f'{disease} Cases by Region',
                    color='Region',
                    text='Cases'
                )
            else:
                # Grouped bar chart for multiple diseases across regions
                fig = px.bar(
                    df,
                    x='Region',
                    y='Cases',
                    color='Disease',
                    title='Disease Distribution by Region',
                    barmode='group'
                )
                
            fig.update_layout(
                xaxis_title='Region',
                yaxis_title='Number of Cases',
                legend_title='Disease'
            )
            
            return fig
        except Exception as e:
            print(f"Error creating comparative visualization: {str(e)}")
            return None