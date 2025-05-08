import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
from dotenv import load_dotenv
import google.generativeai as genai
from bson import ObjectId
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Patient Health Summary API")

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(
    mongo_uri,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = mongo_client[os.getenv("MONGO_DB")]
patients_collection = db["medical_records"]

# Initialize Google Gemini API
genai_api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=genai_api_key)

class PatientIdRequest(BaseModel):
    patient_id: str

@app.post("/generate_summary", response_model=Dict[str, str])
async def generate_summary(request: PatientIdRequest):
    """
    Generate a comprehensive health summary for a patient based on their medical history.
    
    The summary includes:
    - Basic patient information
    - Chronic conditions
    - Allergies
    - Vaccination history
    - Previous visit details
    - Important considerations for treatment
    """
    print(f"Received patient_id: {request.patient_id}")  # Debug log
    
    try:
        # Look for the patient using the 14-digit patient_id field
        patient_data = patients_collection.find_one({"patient_id": request.patient_id})
        
        # Fall back to trying _id (for backward compatibility)
        if not patient_data:
            print(f"Patient not found with patient_id, trying ObjectId...")
            try:
                # Convert string ID to ObjectId for MongoDB query
                object_id = ObjectId(request.patient_id)
                print(f"Converted to ObjectId: {object_id}")
                
                # Fetch patient data from MongoDB using ObjectId
                patient_data = patients_collection.find_one({"_id": object_id})
                print(f"Patient found with ObjectId: {patient_data is not None}")
            except Exception as e:
                print(f"Error converting to ObjectId: {str(e)}")
                # Don't raise an exception yet, we'll handle the not-found case below
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if not patient_data:
        # Extra debug - list collections and sample records
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        # Check if we're using the right collection
        if "medical_records" in collections:
            sample = list(patients_collection.find().limit(1))
            if sample:
                print(f"Sample record structure: patient_id exists: {'patient_id' in sample[0]}")
                print(f"Sample record ID format: {type(sample[0]['_id'])}")
        
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Remove MongoDB ObjectId since it's not JSON serializable
    if isinstance(patient_data["_id"], ObjectId):
        patient_data["_id"] = str(patient_data["_id"])
    
    # Generate summary using Gemini API
    summary = await generate_patient_summary_with_genai(patient_data)
    
    return {"summary": summary}

async def generate_patient_summary_with_genai(patient_data: Dict[str, Any]) -> str:
    """
    Use Google Gemini API to generate a professional health summary based on patient data
    """
    # Create a comprehensive prompt for the AI
    prompt = f"""
    Generate a professional health summary in markdown format for medical professionals based on the following patient data:
    
    {patient_data}
    
    The summary should strictly follow this structure:
    
    # Patient Health Summary
    
    **Patient Name:** [Name]
    **DOB:** [Date of Birth] (Age: [Age] years)
    **Gender:** [Gender]
    **Blood Group:** [Blood Group]
    
    ## Patient Overview
    [Write a comprehensive, clinically-oriented paragraph summarizing the patient's entire health history. Include diagnosis dates for chronic conditions when available. Mention allergies, recurring issues, and recent concerns. This should provide doctors with all critical information at a glance. Use professional medical terminology appropriate for healthcare providers.]
    
    ## Significant Medical History
    - [List chronic conditions and significant medical history]
    
    ## Allergies
    - [List allergies]
    
    ## Vaccination History
    - [List vaccinations with dates]
    
    ## Chronological Visit Summary
    *   **[Date] ([Hospital Name]):**
        - **Diagnosis:** [List diagnoses]
        - **Medications:** [List medications with dosage and duration]
        - **Tests Conducted:** [List tests]
        - **Notes:** [Brief summary of visit notes]
    
    ## Patterns and Key Observations
    - [List observed patterns in health issues, recurring problems]
    - [Any notable progression of conditions]
    
    ## Key Considerations for Future Treatment
    - [Important considerations for future treatment based on history]
    - [Potential medication interactions to be aware of]
    - [Recommendations for follow-up]
    
    The tone should be clinical, professional, and concise. Use medical terminology appropriate for healthcare professionals.
    Ensure ALL sections are included in the response and follow this exact format.
    """
    
    # Call Gemini API with the prompt
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        summary = response.text
        
        # Format check and standardization (but don't add another overview)
        return standardize_summary_format(summary, patient_data)
    except Exception as e:
        # Fallback to template-based summary if AI generation fails
        print(f"Gemini API error: {str(e)}")
        return generate_fallback_summary(patient_data)

def standardize_summary_format(summary: str, patient_data: Dict[str, Any]) -> str:
    """
    Ensure the summary follows the standard format regardless of AI generation
    """
    # Clean up any markdown formatting issues
    summary = summary.replace("```markdown", "").replace("```", "").strip()
    
    # Extract basic patient info for verification
    name = patient_data.get("name", "Unknown")
    dob = patient_data.get("dob", "Unknown")
    gender = patient_data.get("gender", "Not specified")
    blood_group = patient_data.get("blood_group", "Not specified")
    
    # Calculate age
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d")
        age = (datetime.now() - birth_date).days // 365
    except:
        age = "Unknown"
    
    # Check for required sections and add them if missing
    required_sections = [
        ("# Patient Health Summary", "# Patient Health Summary\n\n"),
        ("Patient Name:", f"**Patient Name:** {name}\n"),
        ("DOB:", f"**DOB:** {dob} (Age: {age} years)\n"),
        ("Gender:", f"**Gender:** {gender}\n"),
        ("Blood Group:", f"**Blood Group:** {blood_group}\n\n"),
        ("## Patient Overview", ""),  # Just check if it exists, don't add if missing
        ("## Significant Medical History", "## Significant Medical History\n- " + 
            (", ".join(patient_data.get("chronic_conditions", ["None reported"])) + "\n\n")),
        ("## Allergies", "## Allergies\n- " + 
            (", ".join(patient_data.get("allergies", ["None reported"])) + "\n\n")),
        ("## Vaccination History", "## Vaccination History\n- " + 
            (", ".join([f"{v.get('vaccine', 'Unknown')} ({v.get('date', 'Unknown date')})" 
                      for v in patient_data.get("vaccinations", [])]) or "No vaccination records") + "\n\n"),
        ("## Chronological Visit Summary", "## Chronological Visit Summary\n"),
        ("## Patterns and Key Observations", "## Patterns and Key Observations\n- Recurring health issues should be monitored\n\n"),
        ("## Key Considerations for Future Treatment", "## Key Considerations for Future Treatment\n- Consider chronic conditions during treatment\n- Verify medication compatibility with allergies\n")
    ]
    
    # If a critical section is missing (except Patient Overview which could be added by fallback), rebuild summary with template
    for section_header, section_template in required_sections:
        if section_header not in summary and section_header != "## Patient Overview":
            # This is a major format issue, rebuild the whole summary
            return generate_fallback_summary(patient_data)
    
    # If there are multiple Patient Overview sections, keep only the first one
    first_overview_idx = summary.find("## Patient Overview")
    if first_overview_idx > 0:
        second_overview_idx = summary.find("## Patient Overview", first_overview_idx + 1)
        if second_overview_idx > 0:
            next_section_idx = summary.find("##", second_overview_idx + 1)
            if next_section_idx > 0:
                # Remove the second overview section
                summary = summary[:second_overview_idx] + summary[next_section_idx:]
    
    # Ensure professionally formatted markdown with proper spacing
    summary = re.sub(r'\n{3,}', '\n\n', summary)  # Replace excessive newlines
    
    return summary

def generate_fallback_summary(patient_data: Dict[str, Any]) -> str:
    """
    Generate a standardized template-based summary using the patient data
    """
    # Calculate age from DOB
    try:
        dob = patient_data.get("dob", "Unknown")
        if dob != "Unknown":
            birth_date = datetime.strptime(dob, "%Y-%m-%d")
            age = (datetime.now() - birth_date).days // 365
        else:
            age = "Unknown"
    except:
        age = "Unknown"
    
    # Basic patient info
    name = patient_data.get("name", "Unknown")
    gender = patient_data.get("gender", "Not specified")
    blood_group = patient_data.get("blood_group", "Not specified")
    
    # Chronic conditions and allergies
    chronic_conditions = patient_data.get("chronic_conditions", [])
    allergies = patient_data.get("allergies", [])
    
    # Vaccination history
    vaccinations = patient_data.get("vaccinations", [])
    vaccination_list = [f"- {v.get('vaccine', 'Unknown')} ({v.get('date', 'Unknown date')})" 
                        for v in vaccinations]
    
    # Generate visit summaries
    visits = sorted(patient_data.get("visits", []), 
                   key=lambda x: x.get("date", "1900-01-01"), 
                   reverse=True)
    
    visit_summaries = []
    for visit in visits:
        visit_date = visit.get("date", "Unknown date")
        hospital = visit.get("hospital_name", "Unknown hospital")
        diagnoses = visit.get("diagnosis", ["None"])
        
        medications = []
        for med in visit.get("medications", []):
            med_name = med.get("name", "Unknown")
            dosage = med.get("dosage", "")
            duration = med.get("duration", "")
            medications.append(f"{med_name} {dosage} for {duration}")
        
        tests = visit.get("tests_and_scans", ["None"])
        notes = visit.get("notes", "No additional notes")
        
        visit_summary = f"*   **{visit_date} ({hospital}):**\n"
        visit_summary += f"    - **Diagnosis:** {', '.join(diagnoses)}\n"
        visit_summary += f"    - **Medications:** {', '.join(medications) if medications else 'None prescribed'}\n"
        visit_summary += f"    - **Tests Conducted:** {', '.join(tests)}\n"
        visit_summary += f"    - **Notes:** {notes}\n"
        
        visit_summaries.append(visit_summary)
    
    # Generate patterns and key observations
    patterns = []
    recurring_diagnoses = {}
    
    for visit in visits:
        for diagnosis in visit.get("diagnosis", []):
            recurring_diagnoses[diagnosis] = recurring_diagnoses.get(diagnosis, 0) + 1
    
    for diagnosis, count in recurring_diagnoses.items():
        if count > 1:
            patterns.append(f"- Recurring issue: {diagnosis} ({count} occurrences)")
    
    if not patterns:
        patterns.append("- No recurring patterns identified in available records")
    
    # Generate recommendations
    recommendations = []
    if chronic_conditions:
        recommendations.append(f"- Consider {', '.join(chronic_conditions)} during diagnosis and treatment planning")
    
    if allergies:
        recommendations.append(f"- Verify medication compatibility with allergies to {', '.join(allergies)}")
    
    recommendations.append("- Review previous visit records for treatment effectiveness")
    
    # Generate patient overview
    patient_overview = generate_patient_overview(patient_data, recurring_diagnoses)
    
    # Combine all parts into standardized markdown format
    summary = f"""# Patient Health Summary

**Patient Name:** {name}
**DOB:** {dob} (Age: {age} years)
**Gender:** {gender}
**Blood Group:** {blood_group}

## Patient Overview
{patient_overview}

## Significant Medical History
{("- " + "\\n- ".join(chronic_conditions)) if chronic_conditions else "- None reported"}

## Allergies
{("- " + "\\n- ".join(allergies)) if allergies else "- None reported"}

## Vaccination History
{("\\n".join(vaccination_list)) if vaccination_list else "- No vaccination records"}

## Chronological Visit Summary
{("\\n".join(visit_summaries)) if visit_summaries else "- No previous visits recorded"}

## Patterns and Key Observations
{("\\n".join(patterns))}

## Key Considerations for Future Treatment
{("\\n".join(recommendations))}
"""
    
    return summary

def generate_patient_overview(patient_data: Dict[str, Any], recurring_diagnoses: Dict[str, int] = None) -> str:
    """
    Generate a concise paragraph summarizing the patient's complete health history
    """
    # Extract patient basic info
    name = patient_data.get("name", "Unknown")
    gender = patient_data.get("gender", "Unknown")
    
    # Calculate age
    try:
        dob = patient_data.get("dob", "Unknown")
        if dob != "Unknown":
            birth_date = datetime.strptime(dob, "%Y-%m-%d")
            age = (datetime.now() - birth_date).days // 365
        else:
            age = "Unknown"
    except:
        age = "Unknown"
    
    # Get chronic conditions and allergies
    chronic_conditions = patient_data.get("chronic_conditions", [])
    allergies = patient_data.get("allergies", [])
    
    # Count visits and analyze patterns
    visits = patient_data.get("visits", [])
    visits_count = len(visits)
    
    # Get recent diagnoses if any visits exist
    recent_diagnoses = []
    if visits:
        sorted_visits = sorted(visits, key=lambda x: x.get("date", "1900-01-01"), reverse=True)
        recent_visit = sorted_visits[0]
        recent_diagnoses = recent_visit.get("diagnosis", [])
    
    # If recurring_diagnoses wasn't provided, calculate it
    if recurring_diagnoses is None:
        recurring_diagnoses = {}
        for visit in visits:
            for diagnosis in visit.get("diagnosis", []):
                recurring_diagnoses[diagnosis] = recurring_diagnoses.get(diagnosis, 0) + 1
    
    # Get recurring issues
    recurring_issues = [diagnosis for diagnosis, count in recurring_diagnoses.items() 
                      if count > 1]
    
    # Compose the overview paragraph
    overview = f"{name} is a {age}-year-old {gender}"
    
    if chronic_conditions:
        if len(chronic_conditions) == 1:
            overview += f" with a history of {chronic_conditions[0]}"
        else:
            overview += f" with a history of {', '.join(chronic_conditions[:-1])} and {chronic_conditions[-1]}"
    
    if allergies:
        if len(allergies) == 1:
            overview += f". The patient has a documented allergy to {allergies[0]}"
        else:
            overview += f". The patient has documented allergies to {', '.join(allergies[:-1])} and {allergies[-1]}"
    
    overview += f". Medical records show {visits_count} previous hospital visits"
    
    if recent_diagnoses:
        overview += f", with the most recent diagnosis being {', '.join(recent_diagnoses)}"
    
    if recurring_issues:
        if len(recurring_issues) == 1:
            overview += f". The patient has had recurring episodes of {recurring_issues[0]}"
        else:
            overview += f". The patient has had recurring episodes of {', '.join(recurring_issues[:-1])} and {recurring_issues[-1]}"
    
    # Add any specialist referrals if applicable
    specialist_referrals = []
    for visit in visits:
        if visit.get("referred_to_specialist", False):
            for diagnosis in visit.get("diagnosis", []):
                if diagnosis not in specialist_referrals:
                    specialist_referrals.append(diagnosis)
    
    if specialist_referrals:
        overview += f". The patient has been referred to specialists for {', '.join(specialist_referrals)}"
    
    overview += "."
    
    return overview

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("summary:app", host="0.0.0.0", port=8000, reload=True)

