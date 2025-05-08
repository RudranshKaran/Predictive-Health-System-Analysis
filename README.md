# Predictive Health System Analysis

A comprehensive dashboard for analyzing health data, identifying disease patterns, and predicting health outcomes.

## Features

- **Patient Summary**: View patient visit history and recurring illnesses
- **Biomarker Analysis**: Analyze patient biomarkers and identify abnormalities
- **Regional Analysis**: Visualize health trends across regions
  - Disease prevalence mapping
  - Hotspot detection
  - **NEW: AI-powered root cause analysis and intervention recommendations**
- **Model Performance**: Evaluate predictive model performance

## New Feature: AI-Powered Hotspot Analysis

The system now includes an advanced feature that uses Google's Gemini AI to analyze disease hotspots, infer potential root causes, and recommend targeted interventions.

### How It Works

1. The system identifies disease hotspots using DBSCAN clustering
2. For each hotspot, Gemini AI analyzes:
   - Disease characteristics
   - Regional health indicators
   - Demographic patterns
   - Recent case trends
3. The AI then provides:
   - Potential root causes with confidence levels
   - Recommended interventions with priority levels
   - Urgency assessment
   - Estimated impact of interventions

### Example Use Case

For a cholera outbreak hotspot, the system might identify:
- **Root Cause**: Contaminated water supply (High confidence)
- **Intervention**: Implement water purification systems (High priority)
- **Urgency**: Critical - Rapid spread in densely populated area
- **Impact**: Could reduce new cases by 80% within 2 weeks

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Requests
- Python-dotenv