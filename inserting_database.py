from pymongo import MongoClient
from datetime import datetime, UTC
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import logging
import re
import random
import string

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client[os.getenv("MONGO_DB")]
medical_records = db["medical_records"]  # Single collection for all medical records

# Gemini API configuration
genai.configure(api_key=os.environ.get("GENAI_API_KEY"))

# Function to generate a unique 14-digit patient ID (ABHA-like)
def generate_patient_id():
    # Generate a 14 digit number
    patient_id = ''.join(random.choices(string.digits, k=14))
    return patient_id

# Function to check if a name already exists in the database
def name_exists_in_db(name):
    return medical_records.find_one({"name": name}) is not None

# Get a list of common Indian first and last names to use if needed
first_names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayaan", "Atharva",
    "Muhammad", "Sai", "Arnav", "Aayan", "Krishna", "Ishaan", "Shaurya", "Aryan",
    "Dhruv", "Kabir", "Ritvik", "Aarush", "Kavin", "Darsh", "Rudra", "Pranav",
    "Rishi", "Sahil", "Aahil", "Samarth", "Dhanush", "Aaryan", "Avyaan", "Aradhya",
    "Dhruvin", "Diya", "Saanvi", "Ananya", "Pari", "Aanya", "Aadhya", "Aaradhya",
    "Anvi", "Sara", "Kiara", "Kavya", "Myra", "Shreya", "Siya", "Trisha",
    "Meera", "Mahira", "Aisha", "Eva", "Amyra", "Zara", "Shanaya", "Nisha",
    "Vanya", "Navya", "Prisha", "Sanjana", "Tara", "Aarna", "Anya", "Priya",
    "Riya", "Aditi", "Avni", "Naira"
]

last_names = [
    "Sharma", "Singh", "Kumar", "Shah", "Patel", "Verma", "Gupta", "Joshi",
    "Mehta", "Rao", "Reddy", "Iyer", "Chopra", "Agarwal", "Malhotra", "Sethi",
    "Kapoor", "Chauhan", "Trivedi", "Das", "Mukherjee", "Banerjee", "Chatterjee", "Bhattacharya",
    "Khanna", "Saxena", "Dubey", "Tiwari", "Goel", "Arora", "Chaudhary", "Garg",
    "Bhat", "Nair", "Menon", "Pillai", "Deshmukh", "Jain", "Acharya", "Mahajan",
    "Ganguly", "Sengupta", "Roy", "Pandey", "Shukla", "Bhatia", "Ahuja", "Bajaj",
    "Tambe", "Kaur", "Choudhury", "Basu", "Nayar", "Deshpande", "Patil", "Pawar",
    "Kulkarni", "Kadam", "Mane", "Jadhav", "Gavde", "Thakur", "Shinde", "Mishra"
]

# Bangalore areas with their correct pincodes for diverse address generation
bangalore_areas = [
    {"area": "Koramangala", "streets": ["80 Feet Road", "1st Block", "4th Block", "7th Block", "Sony World Junction", "Forum Mall Road"], "pincode": "560034"},
    {"area": "Indiranagar", "streets": ["12th Main", "100 Feet Road", "Defence Colony", "CMH Road", "Double Road"], "pincode": "560038"},
    {"area": "Jayanagar", "streets": ["4th Block", "9th Block", "30th Cross", "Cool Joint Circle", "Shopping Complex Road"], "pincode": "560041"},
    {"area": "Whitefield", "streets": ["ITPL Main Road", "Hope Farm", "Varthur Main Road", "Graphite India Road", "Whitefield Main Road"], "pincode": "560066"},
    {"area": "Electronic City", "streets": ["Phase 1", "Phase 2", "Neeladri Road", "Infosys Avenue", "Electronic City Main Road"], "pincode": "560100"},
    {"area": "HSR Layout", "streets": ["Sector 1", "Sector 2", "27th Main", "Agara Main Road", "BDA Complex"], "pincode": "560102"},
    {"area": "Bannerghatta", "streets": ["Bannerghatta Main Road", "Bilekahalli", "IIM-B Road", "Hulimavu Gate", "JP Nagar 8th Phase"], "pincode": "560076"},
    {"area": "Malleswaram", "streets": ["8th Cross", "Sampige Road", "Margosa Road", "Malleswaram Circle", "18th Cross"], "pincode": "560003"},
    {"area": "Banashankari", "streets": ["50 Feet Road", "2nd Stage", "3rd Stage", "Kathriguppe Main Road", "Bull Temple Road"], "pincode": "560070"},
    {"area": "Marathahalli", "streets": ["Outer Ring Road", "Kundalahalli Gate", "Varthur Road", "Marathahalli Bridge", "HAL Airport Road"], "pincode": "560037"},
    {"area": "MG Road", "streets": ["M.G. Road", "Brigade Road", "Residency Road", "Church Street", "St. Mark's Road"], "pincode": "560001"},
    {"area": "Basavanagudi", "streets": ["DVG Road", "Bull Temple Road", "Gandhi Bazaar Main Road", "South End Circle", "Tagore Circle"], "pincode": "560004"},
    {"area": "Hebbal", "streets": ["Bellary Road", "Kodigehalli Main Road", "Ring Road", "Manyata Tech Park Road", "Eagle Ridge"], "pincode": "560024"},
    {"area": "Yelahanka", "streets": ["New Town", "Doddaballapur Road", "Rajanukunte", "Yelahanka Old Town", "Kogilu Cross"], "pincode": "560064"},
    {"area": "JP Nagar", "streets": ["15th Cross", "24th Main", "2nd Phase", "5th Phase", "RBI Layout"], "pincode": "560078"},
    {"area": "Kengeri", "streets": ["Mysore Road", "Kommaghatta Road", "Kengeri Satellite Town", "NICE Road Junction", "Kengeri Bus Terminal"], "pincode": "560060"},
    {"area": "Richmond Town", "streets": ["Richmond Road", "Langford Gardens", "Cambridge Road", "Hosur Road", "Wheeler Road"], "pincode": "560025"},
    {"area": "Rajajinagar", "streets": ["1st Block", "4th Block", "West of Chord Road", "Dr. Rajkumar Road", "Basaveshwara Nagar"], "pincode": "560010"},
    {"area": "BTM Layout", "streets": ["16th Main", "29th Main", "7th Cross", "BTM 1st Stage", "BTM 2nd Stage"], "pincode": "560029"},
    {"area": "Sadashivanagar", "streets": ["Sankey Road", "Palace Road", "Cunningham Road", "10th Main", "Bellary Road"], "pincode": "560080"}
]

# Combine system prompt and user prompt
system_prompt = """You are a medical record generator. Generate a SINGLE JSON object containing a complete medical record. 
The output should be VALID JSON without any syntax errors, and follow this structure:
{
    "patient_id": "PATIENT_ID",
    "name": "NAME",
    "dob": "YYYY-MM-DD",
    "gender": "GENDER",
    "blood_group": "BLOOD_GROUP",
    "contact": {
        "phone": "PHONE",
        "email": "EMAIL"
    },
    "address": {
        "street": "STREET_NUMBER STREET_NAME",
        "area": "AREA",
        "city": "Bangalore",
        "district": "Bangalore Urban",
        "state": "Karnataka",
        "pincode": "PINCODE"
    },
    "chronic_conditions": ["CONDITION1", "CONDITION2"],
    "allergies": ["ALLERGY"],
    "vaccinations": [
        {"vaccine": "VACCINE_NAME", "date": "YYYY-MM-DD"}
    ],
    "visits": [
        {
            "date": "YYYY-MM-DD",
            "hospital_name": "HOSPITAL",
            "hospital_id": "ID",
            "diagnosis": ["CONDITION1", "CONDITION2"],
            "medications": [{
                "name": "MEDICINE",
                "dosage": "DOSAGE",
                "duration": "DURATION days"
            }],
            "tests_and_scans": ["TEST1", "TEST2"],
            "notes": "DESCRIPTION",
            "chronic_condition_flagged": true,
            "referred_to_specialist": false
        }
    ]
}

Generate 12-15 visit records in chronological order from 2006 to 2025 with these valid values:
- HOSPITAL: ["Apollo Clinic, Koramangala", "St. John's Hospital", "Manipal Hospital", "Government Health Centre"]
- ID: Corresponding ["hospital_001", "hospital_002", "hospital_003", "hospital_004"]
- CONDITION: ["Common Cold", "Fever", "Anemia", "Fatigue", "Diabetes Type 2", "Hypertension", "Back Pain", "Asthma"]
- MEDICINE: ["Paracetamol", "Ferrous Sulfate", "Metformin", "Amlodipine", "Salbutamol"]
- DOSAGE: Corresponding ["500 mg", "325 mg", "500 mg", "5 mg", "100 mcg"]
- TEST: ["CBC", "Fasting Glucose", "Urine Culture", "TSH", "Chest X-ray", "MRI Brain", "CT Abdomen", "Spirometry"]

For the patient's address, make sure to:
1. Choose from diverse areas across Bangalore using these specific areas and their correct pincodes:
   - Koramangala (560034)
   - Indiranagar (560038)
   - Jayanagar (560041)
   - Whitefield (560066)
   - Electronic City (560100)
   - HSR Layout (560102)
   - Bannerghatta (560076)
   - Malleswaram (560003)
   - Banashankari (560070)
   - Marathahalli (560037)
   - MG Road (560001)
   - Basavanagudi (560004)
   - Hebbal (560024)
   - Yelahanka (560064)
   - JP Nagar (560078)
   - Kengeri (560060)
   - Richmond Town (560025)
   - Rajajinagar (560010)
   - BTM Layout (560029)
   - Sadashivanagar (560080)

2. Create a realistic street address with a house/apartment number and an appropriate street name for the selected area
3. Make sure the area and pincode match exactly as specified above

IMPORTANT:
- Make sure all boolean values are lowercase true/false (not True/False) to ensure JSON validity
- All JSON keys must be in quotes
- Avoid trailing commas in arrays and objects
- Each visit should be complete with all required fields
- The data should be realistic and plausible
- Keep notes concise but informative for each visit
- Make sure medication duration follows format like '3 days' or 'as needed'
- Keep the number of visits between 12-15 to avoid generating too much data
- DO NOT include the patient_id field in your output, I will add it separately
- Generate a unique Indian name that is different from "Priya Sharma"
"""

# Generate a random name suggestion based on our predefined lists
random_first_name = random.choice(first_names)
random_last_name = random.choice(last_names)

user_prompt = f"Generate a complete medical record for a patient in Bangalore with valid JSON syntax. Use a unique Indian name (NOT 'Priya Sharma'). Consider using '{random_first_name} {random_last_name}' or create another unique name."

# Combine prompts
combined_prompt = f"""System: {system_prompt}

User: {user_prompt}

Please respond with ONLY the JSON object, no additional explanations."""

def extract_json_from_text(text):
    """
    Extract a valid JSON object from text that might contain extra content.
    Uses regex to find the JSON pattern and attempts to fix common issues.
    """
    # Find text that looks like JSON (between curly braces, including nested braces)
    json_pattern = r'\{(?:[^{}]|(?R))*\}'
    json_matches = re.findall(r'(\{.*\})', text, re.DOTALL)
    
    if not json_matches:
        raise ValueError("No JSON object found in the text")
        
    # Try each potential JSON match
    for json_text in json_matches:
        # Pre-process to fix common JSON syntax issues
        # Replace True/False with true/false for JSON validity
        json_text = re.sub(r'\bTrue\b', 'true', json_text)
        json_text = re.sub(r'\bFalse\b', 'false', json_text)
        
        # Fix trailing commas in arrays and objects
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Try to parse the JSON
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # If we can't parse this match, continue to the next one
            continue
            
    # If we couldn't parse any of the matches, try using a more aggressive approach
    # Extract the largest text within curly braces
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start >= 0 and end > start:
        potential_json = text[start:end]
        # Apply the same fixes as before
        potential_json = re.sub(r'\bTrue\b', 'true', potential_json)
        potential_json = re.sub(r'\bFalse\b', 'false', potential_json)
        potential_json = re.sub(r',\s*}', '}', potential_json)
        potential_json = re.sub(r',\s*]', ']', potential_json)
        
        # Further cleanup: ensure property names are properly quoted
        # This is a simple approach and might not catch all issues
        potential_json = re.sub(r'(\w+)(?=\s*:)', r'"\1"', potential_json)
        
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError as e:
            # For debugging: print the problematic JSON and the error position
            logger.error(f"JSON error at position {e.pos}: {potential_json[max(0, e.pos-20):min(len(potential_json), e.pos+20)]}")
            
            # Try to fix the specific error
            line_info = str(e).split("line")[1].split("column")
            if len(line_info) > 1:
                line_num = int(line_info[0].strip())
                col_num = int(line_info[1].split()[0].strip())
                
                # Get the problematic line
                lines = potential_json.split('\n')
                if line_num <= len(lines):
                    logger.error(f"Problematic line: {lines[line_num-1]}")
            
            raise ValueError(f"Could not parse JSON: {e}")
    
    raise ValueError("No valid JSON object found in the text")

def generate_unique_record():
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Initialize the Gemini model with generation parameters
            model = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config={"temperature": 0.7, "top_p": 0.8, "top_k": 40}
            )
            
            # Use a single prompt instead of the message array format
            response = model.generate_content(combined_prompt)
            response_content = response.text.strip()
            
            print("Received response from Gemini API")
            
            # Try to extract JSON from the response text
            medical_record = extract_json_from_text(response_content)
            
            # Check if the name already exists in the database
            if "name" in medical_record and name_exists_in_db(medical_record["name"]):
                print(f"Name '{medical_record['name']}' already exists in the database. Trying again...")
                continue
                
            # Generate a unique 14-digit patient ID
            patient_id = generate_patient_id()
            
            # Add the patient_id to the record
            medical_record["patient_id"] = patient_id
            
            # Add creation timestamp
            medical_record["created_at"] = datetime.now(UTC)
            
            # Validate essential fields
            required_fields = ["name", "dob", "gender", "blood_group", "contact", "address", "visits"]
            missing_fields = [field for field in required_fields if field not in medical_record]
            
            if missing_fields:
                logger.error(f"Generated record is missing required fields: {missing_fields}")
                print(f"Missing fields in generated record: {missing_fields}")
                continue
            
            # Ensure the address is diverse and correctly matches our defined areas and pincodes
            if "address" in medical_record:
                address = medical_record["address"]
                # Check if the generated address has valid area and pincode combination
                area_valid = False
                area_name = address.get("area", "")
                pincode = address.get("pincode", "")
                
                # Find the matching area from our predefined list
                matching_area = None
                for bangalore_area in bangalore_areas:
                    if area_name.lower() in bangalore_area["area"].lower() or bangalore_area["area"].lower() in area_name.lower():
                        matching_area = bangalore_area
                        break
                
                # If area exists but pincode doesn't match our predefined data, correct it
                if matching_area:
                    if pincode != matching_area["pincode"]:
                        print(f"Correcting pincode for {area_name} from {pincode} to {matching_area['pincode']}")
                        address["pincode"] = matching_area["pincode"]
                        area_valid = True
                    else:
                        area_valid = True
                
                # If no valid area/pincode combination found, assign a random one from our predefined list
                if not area_valid or not area_name or not pincode:
                    selected_area = random.choice(bangalore_areas)
                    selected_street = random.choice(selected_area["streets"])
                    house_number = random.randint(1, 999)
                    
                    print(f"Replacing address with randomly selected area: {selected_area['area']}")
                    
                    address["street"] = f"{house_number} {selected_street}"
                    address["area"] = selected_area["area"]
                    address["pincode"] = selected_area["pincode"]
                    address["city"] = "Bangalore"
                    address["district"] = "Bangalore Urban"
                    address["state"] = "Karnataka"
            
            # If we got this far, we have a valid record with a unique name
            return medical_record
                
        except Exception as e:
            logger.error(f"Error in attempt {attempt+1}: {str(e)}")
            if attempt == max_attempts - 1:
                raise e
    
    raise ValueError("Failed to generate a unique record after multiple attempts")

response_content = ""  # Initialize response_content to avoid undefined variable error

try:
    # Generate a unique medical record
    medical_record = generate_unique_record()
    
    # Insert complete medical record
    result = medical_records.insert_one(medical_record)
    print(f"Inserted medical record with ID: {result.inserted_id}")
    print(f"Total visits in record: {len(medical_record.get('visits', []))}")
    print(f"Patient name: {medical_record.get('name')}")
    print(f"Patient ID: {medical_record.get('patient_id')}")
    
except json.JSONDecodeError as e:
    logger.error(f"JSON parsing error: {e}")
    # Write the problematic JSON to a file for inspection
    if 'json_text' in locals():
        with open("problematic_json.txt", "w") as f:
            f.write(response_content)
        print(f"JSON parse error: {e}. JSON saved to problematic_json.txt")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    print(f"Error: {e}")
    
    # If there was a response, print a sample for debugging
    if 'response' in locals() and locals()['response']:
        print("\nResponse sample (first 300 chars):")
        if 'response' in locals() and response:
            print(response.text[:300] + ("..." if len(response.text) > 300 else ""))
        else:
            print("No response available to display.")
