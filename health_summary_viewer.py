import streamlit as st
import json
import requests
import re

st.set_page_config(
    page_title="Patient Health Summary Viewer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #4c78a8;
    }
    .patient-info {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .info-label {
        font-weight: 600;
        color: #9e9e9e;
    }
    .info-value {
        font-weight: 600;
        font-size: 1.2rem;
    }
    .visit-summary {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        color: #212529;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .visit-date {
        font-weight: 600;
        color: #0d6efd;
    }
    .diagnosis {
        color: #dc3545;
        font-weight: 600;
    }
    .chronic-condition {
        color: #e41a1c;
        font-weight: 500;
    }
    .allergy {
        color: #ff9da7;
        font-weight: 500;
    }
    .recommendation {
        background-color: #f8f9fa;
        border-left: 3px solid #0d6efd;
        padding: 10px;
        margin-bottom: 10px;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

def extract_markdown_content(response):
    """Extract markdown content from the API response"""
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return response
    
    if isinstance(response, dict) and "summary" in response:
        # Extract the markdown content from the summary
        summary = response["summary"]
        # Remove the markdown code block markers if present
        markdown_content = re.sub(r'^```markdown\n|```$', '', summary, flags=re.MULTILINE)
        return markdown_content
    
    return None

def get_patient_summary(patient_id):
    """Get patient summary from the API"""
    try:
        # The API URL should be configured based on your deployment
        url = "http://localhost:8000/generate_summary"
        payload = {"patient_id": patient_id}
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def parse_patient_info(markdown_content):
    """Parse patient basic information from markdown content"""
    patient_info = {}
    lines = markdown_content.split('\n')
    
    for line in lines:
        if "**Patient Name:**" in line:
            patient_info['name'] = line.split("**Patient Name:**")[1].strip()
        elif "**DOB:**" in line:
            dob_info = line.split("**DOB:**")[1].strip()
            patient_info['dob'] = dob_info.split("(Age:")[0].strip()
            if "(Age:" in dob_info:
                patient_info['age'] = dob_info.split("(Age:")[1].split(")")[0].strip()
        elif "**Gender:**" in line:
            patient_info['gender'] = line.split("**Gender:**")[1].strip()
        elif "**Blood Group:**" in line:
            patient_info['blood_group'] = line.split("**Blood Group:**")[1].strip()
        
        # Stop once we've processed the basic info section
        if line.startswith("## Significant Medical History"):
            break
            
    return patient_info

def display_patient_info(patient_info):
    """Display patient basic information in a nice format"""
    # Add custom CSS for age display
    st.markdown("""
    <style>
        .age-display {
            font-size: 24px;
            font-weight: 700;
            color: white;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .dob-display {
            font-size: 13px;
            color: #FFFFFF;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Patient Name", patient_info.get('name', 'N/A'))
        st.metric("Gender", patient_info.get('gender', 'N/A'))
        # Display DOB as smaller text below gender
        st.markdown(f"<div class='dob-display'>Date of Birth: {patient_info.get('dob', 'N/A')}</div>", unsafe_allow_html=True)
    
    with col2:
        # Display age as the primary metric if available
        if 'age' in patient_info:
            st.metric("Age", f"{patient_info['age']}")
        else:
            # Fallback if age is not available
            st.metric("Age", "N/A")
        
        st.metric("Blood Group", patient_info.get('blood_group', 'N/A'))

def generate_patient_overview(markdown_content):
    """
    Generate a concise paragraph summarizing the patient's entire health history
    """
    # Extract key information from the markdown content
    name = ""
    age = ""
    gender = ""
    chronic_conditions = []
    allergies = []
    visits_count = 0
    recent_diagnoses = []
    recurring_issues = []
    
    sections = re.split(r'## ', markdown_content)
    
    # Extract patient basic info
    if sections[0].strip().startswith('# Patient Health Summary'):
        header_lines = sections[0].strip().split('\n')
        for line in header_lines:
            if "**Patient Name:**" in line:
                name = line.split("**Patient Name:**")[1].strip()
            elif "**DOB:**" in line and "(Age:" in line:
                age = line.split("(Age:")[1].split(")")[0].strip()
            elif "**Gender:**" in line:
                gender = line.split("**Gender:**")[1].strip()
    
    # Extract chronic conditions
    for section in sections[1:]:
        if section.startswith("Significant Medical History"):
            lines = section.split('\n')[1:]  # Skip the header
            for line in lines:
                if line.strip().startswith('- '):
                    condition = line.replace('- ', '').strip()
                    if condition != "None reported":
                        chronic_conditions.append(condition)
        
        # Extract allergies
        elif section.startswith("Allergies"):
            lines = section.split('\n')[1:]  # Skip the header
            for line in lines:
                if line.strip().startswith('- '):
                    allergy = line.replace('- ', '').strip()
                    if allergy != "None reported":
                        allergies.append(allergy)
        
        # Count visits and get recent diagnoses
        elif section.startswith("Chronological Visit Summary"):
            visits = re.findall(r'\*\s+\*\*.*?\*\*', section)
            visits_count = len(visits)
                
            # Extract the most recent diagnosis
            diagnosis_matches = re.findall(r'\*\*Diagnosis:\*\*\s*(.*?)(?=\n)', section)
            if diagnosis_matches:
                recent_diagnoses = [d.strip() for d in diagnosis_matches[0].split(',')]
        
        # Extract recurring issues
        elif section.startswith("Patterns and Key Observations"):
            lines = section.split('\n')[1:]  # Skip the header
            for line in lines:
                if "Recurring issue:" in line or "recurring" in line.lower():
                    recurring_issues.append(line.replace('- ', '').strip())
    
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
        overview += f". Key observations include {recurring_issues[0].lower()}"
    
    overview += "."
    
    return overview

def enhance_markdown_display(markdown_content):
    """
    Enhance the display of markdown content with custom HTML formatting
    """
    html_content = ""
    visit_summary_html = ""
    has_visit_summary = False
    
    # Split the content into sections
    sections = re.split(r'## ', markdown_content)
    
    # Skip processing the header section since we already display this info at the top
    # Just extract patient overview and other sections, not the basic info again
    
    # Process the remaining sections
    for i, section in enumerate(sections[1:], 1):
        if not section.strip():
            continue
            
        # Get section title and content
        section_parts = section.split('\n', 1)
        section_title = section_parts[0].strip()
        section_content = section_parts[1].strip() if len(section_parts) > 1 else ""
        
        # Skip visit summary for now - we'll add it at the end
        if "Visit Summary" in section_title:
            has_visit_summary = True
            visits = re.split(r'\*\s+\*\*', section_content)
            if visits and not visits[0].strip():
                visits = visits[1:]
                
            visit_summary_html += f'<h2 class="section-header">{section_title}</h2>'
            visit_summary_html += '<div>'
            for visit in visits:
                if not visit.strip():
                    continue
                    
                visit_lines = visit.split('\n')
                visit_date_line = f"**{visit_lines[0]}" if visit_lines else ""
                
                visit_summary_html += '<div class="visit-summary">'
                if ":" in visit_date_line:
                    date_part = visit_date_line.split(":")[0]
                    # Remove the extra asterisks from the date display
                    clean_date_part = date_part.replace('**', '')
                    visit_summary_html += f'<div class="visit-date">{clean_date_part}</div>'
                    
                for line in visit_lines[1:]:
                    if "Diagnosis:" in line:
                        diagnosis = line.replace('- **Diagnosis:**', '').strip()
                        visit_summary_html += f'<div><strong>Diagnosis:</strong> <span class="diagnosis">{diagnosis}</span></div>'
                    elif line.strip().startswith('- **'):
                        label = line.split(':**')[0].replace('- **', '').strip()
                        value = line.split(':**')[1].strip() if ':**' in line else ""
                        visit_summary_html += f'<div><strong>{label}:</strong> {value}</div>'
                    elif line.strip():
                        visit_summary_html += f'<div>{line}</div>'
                visit_summary_html += '</div>'
            visit_summary_html += '</div>'
            continue
        
        html_content += f'<h2 class="section-header">{section_title}</h2>'
        
        # Format chronic conditions with special styling
        if "Medical History" in section_title:
            conditions = section_content.split('\n')
            html_content += '<ul>'
            for condition in conditions:
                if condition.strip():
                    condition_text = condition.replace('- ', '').strip()
                    html_content += f'<li class="chronic-condition">{condition_text}</li>'
            html_content += '</ul>'
        
        # Format allergies with special styling
        elif "Allergies" in section_title:
            allergies = section_content.split('\n')
            html_content += '<ul>'
            for allergy in allergies:
                if allergy.strip():
                    allergy_text = allergy.replace('- ', '').strip()
                    html_content += f'<li class="allergy">{allergy_text}</li>'
            html_content += '</ul>'
        
        # Format vaccination history with special styling - each vaccine on a separate line
        elif "Vaccination History" in section_title:
            vaccinations = []
            
            # Handle multiline list format
            if section_content.strip().startswith('- '):
                vaccinations = [vax.replace('- ', '').strip() for vax in section_content.split('\n') if vax.strip()]
            else:
                # Handle single-line format
                vax_line = section_content.strip()
                # Split by common delimiters
                vaccinations = [item.strip() for item in re.split(r' - |, ', vax_line) if item.strip()]
            
            # Add styling for vaccination items - no background color
            html_content += '''
            <style>
                .vaccine-name {
                    font-weight: 600;
                    color: #3498db;
                }
                .vaccine-date {
                    font-size: 0.9em;
                    color: #95a5a6;
                    margin-left: 10px;
                }
                .vax-item {
                    margin-bottom: 8px;
                }
            </style>
            '''
            
            # Create list of vaccination items without background blocks
            html_content += '<div>'
            for vax in vaccinations:
                if vax == "No vaccination records":
                    html_content += f'<div class="vax-item">{vax}</div>'
                    continue
                    
                # Try to extract name and date
                match = re.search(r'(.*?)(?:\s*\(([^)]+)\))?$', vax)
                if match:
                    vax_name = match.group(1).strip()
                    vax_date = match.group(2) if match.group(2) else ""
                    
                    if vax_date:
                        html_content += f'<div class="vax-item"><span class="vaccine-name">{vax_name}</span> <span class="vaccine-date">({vax_date})</span></div>'
                    else:
                        html_content += f'<div class="vax-item"><span class="vaccine-name">{vax_name}</span></div>'
                else:
                    html_content += f'<div class="vax-item">{vax}</div>'
            html_content += '</div>'
            
        # Format recommendations with special styling to match Patterns and Key Observations
        elif "Considerations" in section_title:
            recommendations = section_content.split('\n')
            html_content += '<div style="margin-top: 10px;">'
            for recommendation in recommendations:
                if recommendation.strip():
                    # Remove the bullet point if it exists and add proper styling
                    rec_text = recommendation.replace('- ', '').strip()
                    html_content += f'<div style="margin-bottom: 12px; border-left: 3px solid #4c78a8; padding-left: 10px;">{rec_text}</div>'
            html_content += '</div>'
            
        # Format Patterns and Key Observations with bullet points
        elif "Patterns and Key Observations" in section_title:
            observations = section_content.split('\n')
            html_content += '<div style="margin-top: 10px;">'
            for observation in observations:
                if observation.strip():
                    # Remove the bullet point if it exists and add proper styling
                    obs_text = observation.replace('- ', '').strip()
                    html_content += f'<div style="margin-bottom: 12px; border-left: 3px solid #4c78a8; padding-left: 10px;">{obs_text}</div>'
            html_content += '</div>'
            
        # Default formatting for other sections
        else:
            html_content += f'<div>{section_content}</div>'
    
    # Add the visit summary at the end (if it exists)
    if has_visit_summary:
        html_content += '<div style="margin-top: 40px;"></div>'
    
    return html_content, visit_summary_html if has_visit_summary else None

def main():
    st.title("📊 Patient Health Summary Viewer")
    st.markdown("---")
    
    # Simple and clean interface - just the Patient ID input
    st.subheader("Enter Patient Information")
    
    # Input field for Patient ID with updated placeholder and help text
    patient_id = st.text_input(
        "Patient ID:", 
        placeholder="Enter 14-digit patient ID number",
        help="Enter the 14-digit patient ID (not the MongoDB ObjectID)"
    )
    
    # Generate button
    if st.button("Generate Health Summary", type="primary", use_container_width=True):
        if patient_id:
            # Basic validation for patient ID format
            if patient_id.isdigit() and len(patient_id) == 14:
                with st.spinner("Generating comprehensive health summary..."):
                    response = get_patient_summary(patient_id)
                    if response:
                        markdown_content = extract_markdown_content(response)
                        
                        if markdown_content:
                            st.success("Health summary generated successfully!")
                            st.markdown("---")
                            
                            # Parse and display patient info
                            patient_info = parse_patient_info(markdown_content)
                            display_patient_info(patient_info)
                            
                            st.markdown("# Detailed Patient Health Information", unsafe_allow_html=True)
                            
                            # Display enhanced formatted content
                            enhanced_content, visit_summary_html = enhance_markdown_display(markdown_content)
                            st.markdown(enhanced_content, unsafe_allow_html=True)
                            
                            # Add collapsible visit summary section
                            if visit_summary_html:
                                with st.expander("Chronological Visit Summary"):
                                    st.markdown(visit_summary_html, unsafe_allow_html=True)
                            
                            # Add disclaimer at the bottom
                            st.markdown("""
                            <div style="font-size: 14px; color: #888888; margin-top: 30px; padding: 10px; border-top: 1px solid #444444;">
                            <strong>Disclaimer:</strong> This health summary is generated using artificial intelligence and is intended solely to assist healthcare professionals. 
                            The information provided may not be 100% accurate and should not replace clinical judgment or a thorough review of the patient's medical records. 
                            This tool is meant to supplement, not substitute, professional medical expertise. The final diagnosis and treatment decisions always rest with the attending physician.
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate health summary. Please verify the patient ID and try again.")
            else:
                st.error("Please enter a valid 14-digit patient ID number")
        else:
            st.warning("Please enter a valid patient ID to generate the health summary.")

if __name__ == "__main__":
    main()