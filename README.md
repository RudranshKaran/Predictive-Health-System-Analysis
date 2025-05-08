# Predictive Health System Analysis

A comprehensive healthcare application that generates detailed health summaries for patients based on their medical history.

## Features

- **FastAPI Backend**: Provides an API for generating patient health summaries
- **Streamlit Frontend**: User-friendly interface for viewing health summaries
- **MongoDB Integration**: Stores and retrieves patient medical records
- **AI-Powered Analysis**: Uses AI to generate comprehensive health insights

## Setup

1. Clone repository:
   ```
   git clone https://github.com/yourusername/Predictive-Health-System-Analysis.git
   cd Predictive-Health-System-Analysis
   ```

2. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the `.env.example` template:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file with your actual MongoDB URI and API keys.

## Running the Application

### Backend API

Start the FastAPI server:

```
python summary_api.py
```

Alternatively, you can use uvicorn directly:

```
uvicorn summary_api:app --reload
```

The API will be accessible at `http://localhost:8000`.

### Frontend Application

Start the Streamlit application:

```
streamlit run health_summary_viewer.py
```

The Streamlit interface will be accessible at `http://localhost:8501`.

## API Usage

Send a POST request to `/generate_summary` with the patient ID:

```json
{
  "patient_id": "64a83b1e2f7b7a9aa3e4dc5f"
}
```

The API will return a markdown-formatted summary of the patient's health information.

## API Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
