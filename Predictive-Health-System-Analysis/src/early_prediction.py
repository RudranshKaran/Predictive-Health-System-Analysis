import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("data/datanew.csv")  # Updated to use the new dataset
print(f"Dataset shape: {df.shape}")

# Print the original column names to verify
print("Original column names:", df.columns.tolist())

# First, strip whitespace from column names
df.columns = df.columns.str.strip()

# Print column names after stripping
print("Column names after stripping spaces:", df.columns.tolist())

# Rename columns for consistency and ease of use
column_mapping = {
    'HGB': 'Hemoglobin',
    ' HGB': 'Hemoglobin',  # Handle space before HGB
    'Sex': 'Gender',
    'Sex  ': 'Gender',     # Handle spaces after Sex
    'RBC': 'RBC',
    '  RBC    ': 'RBC',    # Handle spaces around RBC
    'RDW': 'RDW',
    ' RDW    ': 'RDW',     # Handle spaces around RDW
    'Age': 'Age',
    'Age      ': 'Age',    # Handle spaces after Age
    'MCV': 'MCV',
    'MCV  ': 'MCV',        # Handle spaces after MCV
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    ' MCHC  ': 'MCHC'      # Handle spaces around MCHC
}

# Apply the mapping only for columns that exist in the DataFrame
existing_mappings = {col: new_name for col, new_name in column_mapping.items() if col in df.columns}
df = df.rename(columns=existing_mappings)

# Print final column names after renaming
print("Final column names:", df.columns.tolist())

# Convert numeric columns to the correct data types
numeric_columns = ['Hemoglobin', 'RBC', 'Age', 'MCV', 'MCH', 'MCHC', 'RDW']  # Removed 'TLC', 'PLT', 'PCV'
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Also convert Gender to binary numeric (1=male, 0=female)
if 'Gender' in df.columns:
    # First check if it contains text values
    if df['Gender'].dtype == 'object':
        # Map text values to binary
        gender_map = {'M': 1, 'MALE': 1, 'Male': 1, 'male': 1, 'm': 1,
                     'F': 0, 'FEMALE': 0, 'Female': 0, 'female': 0, 'f': 0}
        df['Gender'] = df['Gender'].map(gender_map)
    # Convert to numeric anyway to ensure it's numeric
    df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')

# Print data types after conversion
print("\nData types after conversion:")
print(df.dtypes)

# Clean and prepare data
print("Cleaning and preparing data...")
# Drop any unnamed or unnecessary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values if any
df = df.dropna(subset=['Hemoglobin', 'RBC', 'RDW', 'Age', 'MCV', 'MCH', 'MCHC', 'Gender'])
print(f"Dataset shape after cleaning: {df.shape}")

# Create anemia result column based on WHO guidelines (if not already present)
# WHO defines anemia as hemoglobin < 13 g/dL for men and < 12 g/dL for women
if 'Result' not in df.columns:
    df['Result'] = 0
    male_anemia = (df['Gender'] == 1) & (df['Hemoglobin'] < 13)
    female_anemia = (df['Gender'] == 0) & (df['Hemoglobin'] < 12)
    df.loc[male_anemia | female_anemia, 'Result'] = 1

print(f"Anemia cases: {df['Result'].sum()}, Non-anemia cases: {(df['Result'] == 0).sum()}")

# Create derived features that might capture subtle patterns
print("\n--- Creating derived features for early prediction ---")
# Remove Hemoglobin-based features
if 'Hb_MCH_Ratio' in df.columns:
    df.drop(columns=['Hb_MCH_Ratio', 'Hb_MCHC_Ratio', 'Hb_Gender_Interaction'], inplace=True, errors='ignore')

# Add new derived features without Hb
df['RBC_MCV_Ratio'] = df['RBC'] / df['MCV']
df['RDW_MCV_Ratio'] = df['RDW'] / df['MCV']
df['MCH_MCHC_Ratio'] = df['MCH'] / df['MCHC']
df['Age_RBC_Interaction'] = df['Age'] * df['RBC'] / 100
df['RDW_Age_Interaction'] = df['RDW'] * np.log(df['Age'] + 1)

# Identify borderline cases - where Hemoglobin is normal but close to threshold
# WHO defines anemia as hemoglobin < 13 g/dL for men and < 12 g/dL for women
df['Borderline'] = 0
male_borderline = (df['Gender'] == 1) & (df['Hemoglobin'] >= 13) & (df['Hemoglobin'] < 13.5)
female_borderline = (df['Gender'] == 0) & (df['Hemoglobin'] >= 12) & (df['Hemoglobin'] < 12.5)
df.loc[male_borderline | female_borderline, 'Borderline'] = 1

print(f"Identified {df['Borderline'].sum()} borderline cases that might develop anemia soon")

# Create a target variable for early prediction
# Define "at-risk" individuals as:
# 1. Those who already have anemia, OR
# 2. Those with borderline hemoglobin values
df['Early_Risk'] = np.where((df['Result'] == 1) | (df['Borderline'] == 1), 1, 0)

print(f"Original anemia cases: {df['Result'].sum()}")
print(f"Early risk cases: {df['Early_Risk'].sum()} (including borderline cases)")

# Split into features and target
print("\nPreparing features for model training...")

# Explicitly exclude Hemoglobin from features
features_to_use = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender',
                   'RBC_MCV_Ratio', 'RDW_MCV_Ratio', 'MCH_MCHC_Ratio',
                   'Age_RBC_Interaction', 'RDW_Age_Interaction']
X = df[features_to_use]

# Also drop S. No. if it exists as it's just an identifier
if 'S. No.' in X.columns:
    X = X.drop(columns=['S. No.'])

# Output the final feature list for verification
print(f"\nFeatures used for training: {X.columns.tolist()}")

y = df['Early_Risk']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to handle potential class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Training data shape after SMOTE: {X_train_smote.shape}")

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train and evaluate models
print("\n--- Training Models for Early Anemia Risk Prediction ---")
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    train_score = model.score(X_train_smote, y_train_smote)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel: {name}")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'model/early_predict_cm_{name.replace(" ", "_").lower()}.png')
    
    # Track best model
    if test_score > best_score:
        best_score = test_score
        best_model = model

# Feature importance analysis for the best model
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance for Early Prediction:")
    print(importances)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances)
    plt.title('Feature Importance for Early Anemia Risk Prediction')
    plt.tight_layout()
    plt.savefig('model/early_predict_importance.png')

print("\n--- Creating Risk Stratification System ---")
# Use predicted probabilities to stratify risk
y_probs = best_model.predict_proba(X_test)[:, 1]

# Define risk categories
risk_categories = pd.cut(y_probs, bins=[0, 0.25, 0.5, 0.75, 1.0], 
                       labels=['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk'])

risk_distribution = risk_categories.value_counts()
print("\nRisk distribution in test set:")
print(risk_distribution)

plt.figure(figsize=(10, 6))
sns.countplot(x=risk_categories)
plt.title('Distribution of Anemia Risk Categories')
plt.tight_layout()
plt.savefig('model/early_predict_risk_distribution.png')

print("\nEarly Anemia Risk Prediction completed!")
print("The model can now identify patients at risk of developing anemia")
print("even when their current blood parameters are within normal range.")
print("\nModel outputs:")
print("1. Binary classification (Low Risk/High Risk)")
print("2. Probabilistic risk scores")
print("3. Risk stratification into categories")
print("4. Feature importance analysis showing which combinations of parameters")
print("   indicate future anemia risk")
print("\nNew parameters incorporated:")
print("- RBC (Red Blood Cell count)")
print("- RDW (Red Cell Distribution Width)")
print("- Age")

# Save the best model and scaler for later use
joblib.dump(best_model, 'model/early_anemia_model.pkl')
joblib.dump(scaler, 'model/early_anemia_scaler.pkl')
print("\nModel and scaler saved successfully to model directory!")