# Predictive Health System Analysis

A FastAPI application that generates comprehensive health summaries for patients based on their medical history.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on the `.env.example` template:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file with your actual MongoDB URI and Groq API key.

## Running the Application

Start the FastAPI server:

```
python summary.py
```

Alternatively, you can use uvicorn directly:

```
uvicorn summary:app --reload
```

The API will be accessible at `http://localhost:8000`.

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
```
