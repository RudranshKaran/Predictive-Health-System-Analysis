"""
Gemini AI service for health data analysis.

This module provides integration with Google's Gemini AI to analyze health data,
infer root causes of disease outbreaks, and suggest interventions.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiService:
    """
    Service for interacting with Google's Gemini AI for health data analysis.
    
    This class provides methods to:
    - Analyze disease hotspots
    - Infer potential root causes of outbreaks
    - Generate intervention recommendations
    """
    
    def __init__(self):
        """Initialize the Gemini service with API key from environment variables."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        # Updated to use the correct API version
        
        # Define potential model configurations to try
        self.model_configs = [
            {
                "version": "v1",
                "model": "gemini-1.0-pro",
                "url": "https://generativelanguage.googleapis.com/v1/models/gemini-1.0-pro:generateContent"
            },
            {
                "version": "v1",
                "model": "gemini-pro",
                "url": "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
            },
            {
                "version": "v2",
                "model": "gemini-2.0-flash",
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            }
        ]
        
        # Default to the first configuration
        self.current_model_index = 0
        self.api_url = self.model_configs[self.current_model_index]["url"]
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def analyze_disease_hotspot(self, 
                               disease: str, 
                               region: str, 
                               hotspot_data: Dict[str, Any], 
                               regional_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a disease hotspot to infer root causes and suggest interventions.
        
        Args:
            disease: The name of the disease
            region: The region where the hotspot is located
            hotspot_data: Data about the hotspot (severity, case count, etc.)
            regional_indicators: Health and demographic indicators for the region
            
        Returns:
            Dict containing:
                - root_causes: List of potential root causes with confidence levels
                - interventions: List of recommended interventions
                - urgency_level: Assessed urgency level (low, medium, high, critical)
                - estimated_impact: Estimated impact of interventions
        """
        # Prepare the prompt for Gemini
        prompt = self._create_analysis_prompt(disease, region, hotspot_data, regional_indicators)
        
        # Call Gemini API
        response = self._call_gemini_api(prompt)
        
        # Parse and structure the response
        return self._parse_analysis_response(response)
    
    def _create_analysis_prompt(self, 
                               disease: str, 
                               region: str, 
                               hotspot_data: Dict[str, Any], 
                               regional_indicators: Dict[str, Any]) -> str:
        """
        Create a structured prompt for Gemini to analyze disease hotspots.
        
        Args:
            disease: The name of the disease
            region: The region where the hotspot is located
            hotspot_data: Data about the hotspot
            regional_indicators: Health and demographic indicators for the region
            
        Returns:
            Formatted prompt string
        """
        # Format the hotspot data for the prompt
        hotspot_info = (
            f"Severity: {hotspot_data.get('severity', 'unknown')}\n"
            f"Total cases: {hotspot_data.get('count', 0)}\n"
            f"Recent cases (last 14 days): {hotspot_data.get('recent_cases', 0)}\n"
            f"Geographic radius: {hotspot_data.get('radius', 0):.2f} coordinate units\n"
        )
        
        # Format regional indicators
        indicators_info = ""
        if regional_indicators:
            for category, data in regional_indicators.items():
                if isinstance(data, dict):
                    indicators_info += f"\n{category.replace('_', ' ').title()}:\n"
                    for key, value in data.items():
                        if isinstance(value, dict) and 'mean' in value:
                            indicators_info += f"- {key.replace('_', ' ').title()}: {value['mean']:.2f} (±{value['std']:.2f})\n"
                        else:
                            indicators_info += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        # Create the full prompt
        prompt = f"""
        You are a public health expert analyzing a disease outbreak. Based on the following data about a {disease} hotspot in the {region} region, please:

        1. Identify the most likely root causes of this outbreak
        2. Suggest specific interventions to address these causes
        3. Assess the urgency level (low, medium, high, critical)
        4. Estimate the potential impact of your recommended interventions

        DISEASE HOTSPOT DATA:
        Disease: {disease}
        Region: {region}
        {hotspot_info}

        REGIONAL HEALTH INDICATORS:
        {indicators_info}

        Please format your response as follows:
        
        ROOT CAUSES:
        - [Cause 1]: [Brief explanation] (Confidence: [high/medium/low])
        - [Cause 2]: [Brief explanation] (Confidence: [high/medium/low])
        ...

        RECOMMENDED INTERVENTIONS:
        - [Intervention 1]: [Brief description] (Priority: [high/medium/low])
        - [Intervention 2]: [Brief description] (Priority: [high/medium/low])
        ...

        URGENCY ASSESSMENT:
        [urgency level] - [Brief justification]

        ESTIMATED IMPACT:
        [Brief assessment of potential impact of interventions]
        """
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Gemini API with the given prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            The parsed JSON response from the API
            
        Notes:
            This method will try multiple model configurations if the current one fails.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        # Try each model configuration until one works or all fail
        errors = []
        
        for i in range(len(self.model_configs)):
            # Set the current model configuration
            self.current_model_index = i
            self.api_url = self.model_configs[i]["url"]
            model_name = self.model_configs[i]["model"]
            api_version = self.model_configs[i]["version"]
            
            url = f"{self.api_url}?key={self.api_key}"
            
            try:
                print(f"Trying Gemini model: {model_name} (API version: {api_version})")
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()  # This will raise HTTPError if the response was unsuccessful
           
                # If successful, remember this configuration for future calls
                print(f"Successfully used model: {model_name} (API version: {api_version})")
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                error_msg = f"Error with {model_name} ({api_version}): {str(e)}"
                print(error_msg)
                if hasattr(response, 'text'):
                    print(f"Response: {response.text}")
                errors.append({
                    "model": model_name,
                    "version": api_version,
                    "error": str(e),
                    "details": response.text if hasattr(response, 'text') else "No details available"
                })
            except Exception as e:
                error_msg = f"Unexpected error with {model_name} ({api_version}): {str(e)}"
                print(error_msg)
                errors.append({
                    "model": model_name,
                    "version": api_version,
                    "error": str(e),
                    "details": "Check network connection and API key configuration"
                })
        
        # If all configurations failed, return a structured error response
        return {
            "error": "All Gemini model configurations failed",
            "details": json.dumps(errors, indent=2),
            "tried_models": [config["model"] for config in self.model_configs]
        }
    

    
    def _parse_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the Gemini API response into a structured format.
        
        Args:
            response: The raw API response
            
        Returns:
            Structured analysis results
        """
        # Check if the response contains an error
        if "error" in response:
            return {
                "error": response.get("error"),
                "details": response.get("details", "No details available"),
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }
            
        try:
            # Extract the text content from the response
            content = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Initialize the result structure
            result = {
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }
            
            # Parse root causes
            if "ROOT CAUSES:" in content:
                causes_section = content.split("ROOT CAUSES:")[1].split("RECOMMENDED INTERVENTIONS:")[0]
                causes_lines = [line.strip() for line in causes_section.strip().split("\n") if line.strip().startswith("-")]
                
                for cause in causes_lines:
                    # Extract confidence level
                    confidence = "medium"  # Default
                    if "Confidence: high" in cause.lower():
                        confidence = "high"
                    elif "Confidence: low" in cause.lower():
                        confidence = "low"
                    
                    # Extract the cause text (remove the leading dash and confidence part)
                    cause_text = cause[1:].strip()
                    if "(Confidence:" in cause_text:
                        cause_text = cause_text.split("(Confidence:")[0].strip()
                    
                    result["root_causes"].append({
                        "cause": cause_text,
                        "confidence": confidence
                    })
            
            # Parse interventions
            if "RECOMMENDED INTERVENTIONS:" in content:
                interventions_section = content.split("RECOMMENDED INTERVENTIONS:")[1].split("URGENCY ASSESSMENT:")[0]
                intervention_lines = [line.strip() for line in interventions_section.strip().split("\n") if line.strip().startswith("-")]
                
                for intervention in intervention_lines:
                    # Extract priority level
                    priority = "medium"  # Default
                    if "Priority: high" in intervention.lower():
                        priority = "high"
                    elif "Priority: low" in intervention.lower():
                        priority = "low"
                    
                    # Extract the intervention text
                    intervention_text = intervention[1:].strip()
                    if "(Priority:" in intervention_text:
                        intervention_text = intervention_text.split("(Priority:")[0].strip()
                    
                    result["interventions"].append({
                        "action": intervention_text,
                        "priority": priority
                    })
            
            # Parse urgency level
            if "URGENCY ASSESSMENT:" in content:
                urgency_section = content.split("URGENCY ASSESSMENT:")[1].split("ESTIMATED IMPACT:")[0].strip()
                
                # Extract the urgency level (first word)
                urgency_words = urgency_section.split(" - ")[0].strip().lower()
                if "critical" in urgency_words:
                    result["urgency_level"] = "critical"
                elif "high" in urgency_words:
                    result["urgency_level"] = "high"
                elif "medium" in urgency_words:
                    result["urgency_level"] = "medium"
                elif "low" in urgency_words:
                    result["urgency_level"] = "low"
                
                # Extract justification
                if " - " in urgency_section:
                    result["urgency_justification"] = urgency_section.split(" - ")[1].strip()
            
            # Parse estimated impact
            if "ESTIMATED IMPACT:" in content:
                impact_section = content.split("ESTIMATED IMPACT:")[1].strip()
                result["estimated_impact"] = impact_section
            
            return result
            
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            print(f"Raw response: {json.dumps(response, indent=2)}")
            return {
                "error": f"Failed to parse response: {str(e)}",
                "root_causes": [],
                "interventions": [],
                "urgency_level": "unknown",
                "estimated_impact": ""
            }