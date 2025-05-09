import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os

def show_landing_page():
    """Display a landing page based on the provided PDF design"""
    
    # Remove padding at the top
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            .main > div {
                padding-left: 2rem;
                padding-right: 2rem;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Arial', sans-serif;
            }
            .highlight-text {
                background-color: #f0f8ff;
                padding: 0.2rem 0.5rem;
                border-radius: 5px;
                font-weight: bold;
                color: #205493;
            }
            .badge {
                background-color: #205493;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                margin-right: 0.5rem;
                font-size: 0.8rem;
            }
            .card {
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                background: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .btn-primary {
                background-color: #205493;
                color: white;
                padding: 0.6rem 1.2rem;
                font-weight: bold;
                border-radius: 5px;
                text-align: center;
                display: inline-block;
                margin: 0.5rem 0;
                text-decoration: none;
            }
            .btn-secondary {
                background-color: #757575;
                color: white;
                padding: 0.6rem 1.2rem;
                font-weight: bold;
                border-radius: 5px;
                text-align: center;
                display: inline-block;
                margin: 0.5rem 0;
                text-decoration: none;
            }
            .footer {
                text-align: center;
                padding: 1rem;
                color: #757575;
                font-size: 0.8rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: transparent;
            }
            .css-1cpxqw2 {
                font-weight: bold !important;
                font-size: 1.25rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='font-size: 2.5rem; color: #205493;'>Predictive Health System Hackathon</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #757575; margin-top: -1rem;'>Building the future of healthcare analytics</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1rem;">
            <span class="badge">May 9-11, 2025</span>
            <span class="badge">Online & In-Person</span>
            <span class="badge">$10,000 in Prizes</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <a href="#" class="btn-primary">Register Now</a>
            <a href="#" class="btn-secondary" style="margin-left: 1rem;">Learn More</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Logo/Image placeholder
        st.image("https://via.placeholder.com/300x200?text=Health+Tech+Logo", width=300)
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # About the Hackathon
    st.markdown("<h2>About the Hackathon</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Join healthcare professionals, data scientists, and developers in a weekend-long challenge to
    build innovative solutions using our Predictive Health System API. Leverage real-world anonymized 
    health data to create predictive models that could transform patient care.
    
    Our platform provides:
    - Early disease prediction using biomarkers
    - Regional health trend analysis
    - Personalized health summaries
    - Clinical decision support tools
    """)
    
    # Challenges Section
    st.markdown("<h2>Challenges</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔬 Biomarker Analysis</h3>
            <p>Develop algorithms to identify early disease markers from lab results before traditional diagnostic thresholds are met.</p>
            <p><strong>Focus areas:</strong> Anemia, Diabetes, Cardiovascular</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🗺️ Regional Trends</h3>
            <p>Create visualization tools that analyze regional health patterns and correlate them with environmental or demographic factors.</p>
            <p><strong>Focus areas:</strong> Geographic heat maps, Time-series analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>📋 Health Summaries</h3>
            <p>Build intelligent systems that generate personalized health summaries for patients and providers based on available data.</p>
            <p><strong>Focus areas:</strong> NLP, Risk stratification</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Timeline 
    st.markdown("<h2>Event Timeline</h2>", unsafe_allow_html=True)
    
    timeline_data = {
        "Event": ["Kickoff & API Introduction", "Team Formation", "Coding Period", "Mentoring Sessions", "Project Submissions", "Demos & Judging", "Awards Ceremony"],
        "Time": ["May 9, 9:00 AM", "May 9, 11:00 AM", "May 9-11", "Throughout Event", "May 11, 12:00 PM", "May 11, 2:00 PM", "May 11, 5:00 PM"],
        "Duration": ["2 hours", "1 hour", "48 hours", "Scheduled", "Deadline", "3 hours", "1 hour"]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create a styled HTML table for the timeline
    html_table = """
    <table style="width:100%; border-collapse: collapse; margin: 1rem 0;">
        <tr style="background-color: #205493; color: white;">
            <th style="padding: 10px; text-align: left;">Event</th>
            <th style="padding: 10px; text-align: left;">Time</th>
            <th style="padding: 10px; text-align: left;">Duration</th>
        </tr>
    """
    
    for i, row in df_timeline.iterrows():
        bg_color = "#f0f8ff" if i % 2 == 0 else "white"
        html_table += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">{row['Event']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">{row['Time']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">{row['Duration']}</td>
        </tr>
        """
    
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Prizes
    st.markdown("<h2>Prizes</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="border-left: 5px solid gold; text-align: center;">
            <h3>🥇 First Place</h3>
            <h4 style="color: #205493;">$5,000</h4>
            <p>Plus mentorship from industry leaders and potential pilot opportunity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="border-left: 5px solid silver; text-align: center;">
            <h3>🥈 Second Place</h3>
            <h4 style="color: #205493;">$3,000</h4>
            <p>Plus continued development support and industry exposure</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="border-left: 5px solid #cd7f32; text-align: center;">
            <h3>🥉 Third Place</h3>
            <h4 style="color: #205493;">$1,500</h4>
            <p>Plus recognition and networking opportunities</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Category winners
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem;">
        <h4>Category Winners</h4>
        <p>Best Technical Innovation: $500 | Most User-Friendly: $500 | Most Impactful: $500</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Judges and Mentors
    st.markdown("<h2>Judges & Mentors</h2>", unsafe_allow_html=True)
    
    judges = [
        {"name": "Dr. Sarah Chen", "title": "Chief Medical Information Officer, General Hospital", "image": "https://via.placeholder.com/150"},
        {"name": "Michael Roberts", "title": "AI Research Director, Health Innovations Inc.", "image": "https://via.placeholder.com/150"},
        {"name": "Dr. James Wilson", "title": "Professor of Biomedical Informatics, University", "image": "https://via.placeholder.com/150"},
        {"name": "Aisha Patel", "title": "Health Tech Entrepreneur & Investor", "image": "https://via.placeholder.com/150"}
    ]
    
    cols = st.columns(4)
    for i, judge in enumerate(judges):
        with cols[i]:
            st.image(judge["image"], width=150)
            st.markdown(f"""
            <div style="text-align: center;">
                <h4 style="margin-bottom: 0;">{judge["name"]}</h4>
                <p style="font-size: 0.8rem; color: #757575;">{judge["title"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Resources Section
    st.markdown("<h2>Resources</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>API Documentation</h3>
            <p>Get started with our comprehensive API docs:</p>
            <ul>
                <li>Biomarker Analysis API</li>
                <li>Regional Health Trends API</li>
                <li>Health Summary Generator API</li>
            </ul>
            <a href="#" class="btn-primary">View Documentation</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Sample Projects</h3>
            <p>Explore sample implementations to jumpstart your project:</p>
            <ul>
                <li>Early Anemia Detection Dashboard</li>
                <li>Regional Health Disparities Map</li>
                <li>Patient Risk Stratification Tool</li>
            </ul>
            <a href="#" class="btn-primary">View Examples</a>
        </div>
        """, unsafe_allow_html=True)
    
    # FAQ Section
    st.markdown("<h2>Frequently Asked Questions</h2>", unsafe_allow_html=True)
    
    with st.expander("Who can participate?"):
        st.write("""
        Anyone interested in healthcare technology can participate! We welcome:
        - Data scientists and ML engineers
        - Software developers
        - Healthcare professionals
        - UX/UI designers
        - Students in relevant fields
        
        Teams can have up to 5 members.
        """)
    
    with st.expander("What skills are needed?"):
        st.write("""
        Teams benefit from diverse skill sets, including:
        - Programming (Python, R, JavaScript)
        - Data analysis and visualization
        - Machine learning
        - Healthcare domain knowledge
        - UX/UI design
        
        Don't have a complete team? Join our team formation session!
        """)
    
    with st.expander("Is the data real patient data?"):
        st.write("""
        No, we use synthetic but realistic health data that mimics real-world patterns. All data provided is compliant with privacy regulations and has been generated specifically for this hackathon.
        """)
    
    with st.expander("How will projects be judged?"):
        st.write("""
        Projects will be evaluated based on:
        - Technical innovation (30%)
        - Accuracy and performance (25%)
        - Potential clinical impact (25%)
        - User experience and design (15%)
        - Presentation quality (5%)
        
        Each team will have 5 minutes to present plus 2 minutes for Q&A.
        """)
    
    # Contact
    st.markdown("<h2>Contact Us</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <p>Have questions about the hackathon? Reach out to us!</p>
        <p><strong>Email:</strong> hackathon@healthsystem.org | <strong>Phone:</strong> (555) 123-4567</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Registration CTA
    st.markdown("""
    <div style="text-align: center; background-color: #205493; color: white; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: white;">Ready to build the future of healthcare?</h2>
        <p style="font-size: 1.2rem;">Teams are limited to 100 participants, so register soon!</p>
        <a href="#" class="btn-primary" style="background-color: white; color: #205493; font-size: 1.2rem; padding: 0.8rem 1.5rem; margin-top: 1rem;">Register Now</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Predictive Health System Hackathon | Sponsored by Health Innovations Inc. | Privacy Policy | Terms of Service</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_landing_page()
