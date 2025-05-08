from pymongo import MongoClient
from datetime import datetime, UTC
import os
from groq import Groq
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
MONGO_URI = "mongodb+srv://tutorial_db:tutorial_db_password@predictiveanalysis.x1ye35v.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["Health_Summary"]
medical_records = db["medical_records"]  # Single collection for all medical records

# Groq client configuration
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "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")
)

system_prompt = """You are a medical record generator. Generate a SINGLE JSON object containing a complete medical record. The output should follow this exact structure:
{{
    "name": "NAME",
    "dob": "YYYY-MM-DD",
    "gender": "GENDER",
    "blood_group": "BLOOD_GROUP",
    "contact": {{
        "phone": "PHONE",
        "email": "EMAIL"
    }},
    "address": {{
        "street": "STREET",
        "city": "Bangalore",
        "district": "Bangalore Urban",
        "state": "Karnataka",
        "pincode": "560034"
    }},
    "chronic_conditions": ["CONDITION1", "CONDITION2"],
    "allergies": ["ALLERGY"],
    "vaccinations": [
        {{"vaccine": "VACCINE_NAME", "date": "YYYY-MM-DD"}}
    ],
    "visits": [
        {{
            "date": "YYYY-MM-DD",
            "hospital_name": "HOSPITAL",
            "hospital_id": "ID",
            "diagnosis": ["CONDITION1", "CONDITION2"],
            "medications": [{{
                "name": "MEDICINE",
                "dosage": "DOSAGE",
                "duration": "DURATION days"
            }}],
            "tests_and_scans": ["TEST1", "TEST2"],
            "notes": "DESCRIPTION",
            "chronic_condition_flagged": boolean,
            "referred_to_specialist": boolean
        }}
    ]
}}

Generate at least 60 visit records in chronological order from 2006 to 2025 with these valid values:
- HOSPITAL: ["Apollo Clinic, Koramangala", "St. John's Hospital", "Manipal Hospital", "Government Health Centre"]
- ID: Corresponding ["hospital_001", "hospital_002", "hospital_003", "hospital_004"]
- CONDITION: ["Common Cold", "Fever", "Anemia", "Fatigue", "Diabetes Type 2", "Hypertension", "Back Pain", "Asthma"]
- MEDICINE: ["Paracetamol", "Ferrous Sulfate", "Metformin", "Amlodipine"]
- DOSAGE: Corresponding ["500 mg", "325 mg", "500 mg", "5 mg"]
- TEST: ["CBC", "Fasting Glucose", "Urine Culture", "TSH", "Chest X-ray", "MRI Brain", "CT Abdomen"]

REMEBMER:
- The data should be realistic and plausible.
- The data should range from birth to present day.
- The data should be in chronological order.
- Fields like test_and_scans, medications, and notes should be filled with realistic values and can be empty.
"""

user_prompt = "Generate a complete medical record for a patient in Bangalore"

try:
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="gemma2-9b-it",
        temperature=0.7
    )

    response_content = chat_completion.choices[0].message.content.strip()
    json_start = response_content.find('{')
    json_end = response_content.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_str = response_content[json_start:json_end]
        medical_record = json.loads(json_str)
        medical_record["created_at"] = datetime.now(UTC)
        
        # Insert complete medical record
        result = medical_records.insert_one(medical_record)
        print(f"Inserted medical record with ID: {result.inserted_id}")
        print(f"Total visits in record: {len(medical_record.get('visits', []))}")
    else:
        logger.error("No JSON object found in response")

except json.JSONDecodeError as e:
    logger.error(f"JSON parsing error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
