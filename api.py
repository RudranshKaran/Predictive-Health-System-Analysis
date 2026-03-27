from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import requests
from dotenv import load_dotenv

from src.patient_summary import PatientSummary
from src.biomarker_analysis import BiomarkerAnalysis
from src.regional_analysis import RegionalAnalysis

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path of the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize modules and load data
try:
    patient_module = PatientSummary()
    patient_data_path = os.path.join(BASE_DIR, "data", "patient_data.csv")
    patient_module.load_data(patient_data_path)

    biomarker_module = BiomarkerAnalysis()
    biomarker_data_path = os.path.join(BASE_DIR, "data", "biomarker_data.csv")
    reference_ranges_path = os.path.join(BASE_DIR, "data", "reference_ranges.csv")
    biomarker_module.load_data(biomarker_data_path, reference_ranges_path)

    regional_module = RegionalAnalysis()
    regional_data_path = os.path.join(BASE_DIR, "data", "regional_data.csv")
    regional_module.load_data(regional_data_path)
    
    # Temporarily comment out MedGemma model loading to resolve CORS issue
    # medgemma_tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b")
    # medgemma_model = AutoModelForCausalLM.from_pretrained("google/medgemma-27b")
    medgemma_tokenizer = None
    medgemma_model = None

    # Load DDI library
    ddi_library_path = os.path.join(BASE_DIR, "data", "ddi_library.json")
    with open(ddi_library_path, 'r') as f:
        ddi_library = json.load(f)

except Exception as e:
    print(f"Error loading data or models: {str(e)}")
    # Handle error appropriately
    patient_module = None
    biomarker_module = None
    regional_module = None
    medgemma_tokenizer = None
    medgemma_model = None
    ddi_library = []

@app.route('/')
def home():
    return "Predictive Health System API is running."

@app.route('/patient_summary/<patient_id>', methods=['GET'])
def get_patient_summary(patient_id):
    if not patient_module:
        return jsonify({"error": "Patient module not initialized"}), 500
    
    try:
        timeline = patient_module.get_visit_timeline(patient_id)
        if not timeline:
            return jsonify({"error": "No data found for this patient ID"}), 404
        
        recurring_illnesses = patient_module.get_recurring_illnesses(patient_id)
        
        summary_data = {
            "visit_timeline": timeline,
            "recurring_illnesses": recurring_illnesses
        }
        
        return jsonify(summary_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/biomarker_analysis/<patient_id>', methods=['GET'])
def get_biomarker_analysis(patient_id):
    if not biomarker_module:
        return jsonify({"error": "Biomarker module not initialized"}), 500

    try:
        analysis = biomarker_module.analyze_biomarkers(patient_id)
        if not analysis:
            return jsonify({"error": "No biomarker data found for this patient ID"}), 404
            
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/regional_trends/<region>', methods=['GET'])
def get_regional_trends(region):
    if not regional_module:
        return jsonify({"error": "Regional module not initialized"}), 500

    try:
        analysis = regional_module.analyze_regional_patterns(region)
        if not analysis:
            return jsonify({"error": "No data available for selected region"}), 404
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ddi_check', methods=['POST'])
def ddi_check():
    data = request.get_json()
    if not data or 'current_medications' not in data or 'new_medications' not in data:
        return jsonify({"error": "Invalid request body"}), 400

    current_meds = [med.lower() for med in data['current_medications']]
    new_meds = [med.lower() for med in data['new_medications']]
    interactions = []

    all_meds = current_meds + new_meds

    for i in range(len(all_meds)):
        for j in range(i + 1, len(all_meds)):
            med1 = all_meds[i]
            med2 = all_meds[j]
            for interaction in ddi_library:
                drugA = interaction['drugA'].lower()
                drugB = interaction['drugB'].lower()
                if (med1 == drugA and med2 == drugB) or \
                   (med1 == drugB and med2 == drugA):
                    interactions.append(interaction)

    return jsonify({"interactions": interactions})

@app.route('/medgemma_analyze', methods=['POST'])
def medgemma_analyze():
    # This endpoint is intentionally left non-functional to test the fallback.
    # It simulates the MedGemma service being unavailable.
    return jsonify({"error": "MedGemma service is currently unavailable."}), 503

@app.route('/gemini_analyze', methods=['POST'])
def gemini_analyze():
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "GOOGLE_API_key not found in environment variables."}), 500

    data = request.get_json()
    if not data or 'text' not in data or 'region' not in data:
        return jsonify({"error": "Request must include 'text' and 'region'"}), 400

    input_text = data['text']
    region = data['region']
    current_medications = data.get('current_medications', [])
    
    # Fetch regional trend data
    regional_context = "No regional data available."
    if regional_module:
        try:
            trends = regional_module.analyze_regional_patterns(region)
            if trends and trends.get('top_diseases'):
                top_diseases = ', '.join(trends['top_diseases'])
                regional_context = f"Current health trends in {region}: There is a high prevalence of {top_diseases}. Please take this into consideration."
        except Exception as e:
            print(f"Could not fetch regional data: {e}")

    # Format current medications for the prompt
    medication_context = "The patient is not currently taking any medications."
    if current_medications:
        medication_context = f"The patient is currently taking: {', '.join(current_medications)}."


    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"

    system_prompt = f"""
    You are an expert clinical assistant AI writing suggestions for a qualified doctor. Your task is to extract structured information from clinical notes and provide a differential diagnosis that is heavily biased by regional health trends. Your tone should be professional, collaborative, and suggestive, not prescriptive.
    
    PATIENT CONTEXT: {medication_context}
    REGIONAL CONTEXT: {regional_context}
    
    Analyze the provided clinical text. Based on the clinical notes, patient context, AND regional context, return a JSON object.
    ALWAYS include all keys ("symptoms", "diagnosis", "medications", "missingInfo").
    
    JSON Structure:
    {{
      "symptoms": [{{"value": "Symptom description", "confidence": 0.95}}],
      "diagnosis": [
          {{"description": "Most Likely Diagnosis", "code": "J06.9", "suggestion": true, "confidence": 0.85}},
          {{"description": "Alternative Diagnosis to Consider", "code": "A90", "suggestion": false, "confidence": 0.70}}
       ],
      "medications": [{{"name": "Paracetamol", "dosage": "500mg", "frequency": "TDS", "confidence": 0.99}}],
      "missingInfo": ["Patient's blood pressure was not recorded. Request vitals."]
    }}

    Instructions:
    - Your top priority is to use the REGIONAL CONTEXT to bias your diagnosis. For example, if the notes say "fever, headache" and the regional context says "high prevalence of Dengue", Dengue MUST be a top differential diagnosis, even if the symptoms also fit other viral fevers.
    - For "diagnosis", generate a list of potential diagnoses, with the most likely one first. The most likely diagnosis should have `"suggestion": true`. All other possibilities should have `"suggestion": false`.
    - CRITICAL SAFETY INSTRUCTION: Before suggesting any medication, you MUST consider the patient's current medications ({medication_context}). If a common treatment interacts with them, note the risk and suggest a safer alternative.
    - Based on your "Most Likely Diagnosis", suggest a potential first-line medication to consider, unless a treatment is already mentioned. For common symptoms (e.g., fever, cough), suggesting a symptomatic treatment is appropriate. Frame it as a suggestion (e.g., "Consider Paracetamol...").
    - If the text provides absolutely no clinical information, return a diagnosis of `[{{"description": "No clinical information provided", "code": "N/A", "suggestion": false, "confidence": 0.0}}]`.
    - Confidence score (0.0 to 1.0) must be based on how certain you are.
    - Ensure the output is a single, clean, valid JSON object. All outputs are suggestions for clinical review, not direct medical advice.
    """

    payload = {
        "contents": [{"parts": [{"text": input_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
        }
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # The response should be a JSON object as requested
        result = response.json()
        
        # The actual content is nested inside the response
        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        # Parse the JSON string from the 'text' field
        structured_response = json.loads(content)
        
        return jsonify(structured_response)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return jsonify({"error": f"Failed to parse API response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "GOOGLE_API_KEY not found"}), 500

    patient_data = request.get_json()
    if not patient_data:
        return jsonify({"error": "No patient data provided"}), 400

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"

    system_prompt = """
    You are an expert clinical summarization AI. Your task is to synthesize a patient's entire medical record into a concise, actionable summary for a busy clinician.
    
    Instructions:
    1.  **Synthesize, Don't Just List:** Do not just list the patient's conditions. Connect the dots. For example, instead of "Has hypertension. Takes Amlodipine.", write "Manages hypertension with Amlodipine."
    2.  **Prioritize Critical Information:** Start with the most important clinical facts. Red flag alerts (like critical allergies) and major chronic conditions should be mentioned first.
    3.  **Incorporate All Data:** Your summary MUST be informed by all sections of the patient data provided: age, gender, preExistingConditions, allergies, currentMedications, recurringIllnesses, and the visitHistory.
    4.  **Keep it Concise:** The final summary should be a single, dense paragraph, no more than 3-4 sentences long.
    """

    # We send the whole patient object to give the AI complete context.
    payload = {
        "contents": [{
            "parts": [{"text": f"Generate a clinical summary for the following patient: {json.dumps(patient_data)}"}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        summary_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not generate summary.')
        
        return jsonify({"summary": summary_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Failed to parse API response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
