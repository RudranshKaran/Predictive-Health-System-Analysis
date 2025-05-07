import os
import sys
import subprocess

# Change to parent directory to avoid any potential numpy directory conflict
os.chdir("..")

# Run streamlit with app.py in the original directory
app_path = os.path.join("Predictive-Health-System-Analysis", "app.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])