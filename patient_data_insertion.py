from pymongo import MongoClient
from datetime import datetime

# Replace with your actual MongoDB URI (use Atlas URI or "mongodb://localhost:27017/")
MONGO_URI = "mongodb+srv://tutorial_db:tutorial_db_password@predictiveanalysis.x1ye35v.mongodb.net/"
client = MongoClient(MONGO_URI)

# Select the database and collection
db = client["Health_Summary"]
patients_col = db["patients"]

# Sample patient document
patient_data = {
    "name": "Ravi Kumar",
    "dob": "2000-07-18",
    "gender": "Male",
    "blood_group": "B+",
    "contact": {
        "phone": "+91-9876543210",
        "email": "ravi@example.com"
    },
    "address": {
        "street": "123 MG Road",
        "city": "Bangalore",
        "district": "Bangalore Urban",
        "state": "Karnataka",
        "pincode": "560034"
    },
    "chronic_conditions": ["Diabetes", "Hypertension"],
    "allergies": ["Penicillin"],
    "vaccinations": [
        {"vaccine": "COVID-19", "date": "2021-06-15"},
        {"vaccine": "Tetanus", "date": "2019-11-02"}
    ],
    "created_at": datetime.utcnow()
}

# Insert into the patients collection
result = patients_col.insert_one(patient_data)

print(f"Inserted patient with _id: {result.inserted_id}")
