import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv("data/datanew.csv")  # Using the new dataset
print(f"Dataset shape: {df.shape}")

# Print column names to verify
print("Original columns:", df.columns.tolist())

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()
print("Cleaned columns:", df.columns.tolist())

# Print data types to understand column format issues
print("\nData types before conversion:")
print(df.dtypes)

# Convert columns that should be numeric to float
numeric_columns = ['Age', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT /mm3', 'HGB']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

# Handle missing values
print(f"\nMissing values before imputation:\n{df.isnull().sum()}")
df = df.dropna()  # Or use imputation strategies
print(f"Dataset shape after handling missing values: {df.shape}")

# Rename columns for consistency and ease of use
df = df.rename(columns={
    'HGB': 'Hemoglobin',
    'Sex': 'Gender',
    'RBC': 'RBC',
    'RDW': 'RDW',
    'Age': 'Age',
    'MCV': 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'TLC': 'TLC',
    'PLT /mm3': 'PLT'
})

# Create anemia result column based on WHO guidelines (if not already present)
if 'Result' not in df.columns:
    print("Creating Result column based on WHO guidelines")
    df['Result'] = 0
    # Convert Gender column to categorical if it's not already
    if df['Gender'].dtype == object:
        print("Converting Gender from text to binary")
        gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 'm': 1, 'f': 0}
        df['Gender'] = df['Gender'].map(gender_map).astype(float)
    
    male_anemia = (df['Gender'] == 1) & (df['Hemoglobin'] < 13)
    female_anemia = (df['Gender'] == 0) & (df['Hemoglobin'] < 12)
    df.loc[male_anemia | female_anemia, 'Result'] = 1

print(f"Anemia cases: {df['Result'].sum()}, Non-anemia cases: {(df['Result'] == 0).sum()}")

# Create a more realistic dataset by adding noise
print("\n--- Creating more realistic dataset with noise ---")
df_noise = df.copy()

# Add noise to numerical features to simulate measurement variability
# Use medical device accuracy ranges as a guide
noise_factors = {
    'Hemoglobin': 0.03,  # 3% measurement error is typical for hemoglobin tests
    'MCH': 0.02,
    'MCHC': 0.015,
    'MCV': 0.025,
    'RBC': 0.02,
    'RDW': 0.015
}

# Add controlled noise to each feature
for col, factor in noise_factors.items():
    if col in df_noise.columns:
        noise = np.random.normal(0, df_noise[col].std() * factor, size=len(df_noise))
        df_noise[col] = df_noise[col] + noise

print("Noise added to create more realistic data")

# Create derived features that might be clinically relevant
df_noise['RBC_Hb_Ratio'] = df_noise['RBC'] / df_noise['Hemoglobin']
df_noise['RDW_MCV_Ratio'] = df_noise['RDW'] / df_noise['MCV']
df_noise['Age_Hb_Interaction'] = df_noise['Age'] * df_noise['Hemoglobin'] / 100  # Scaling to avoid large values

# Remove Hemoglobin-based features
if 'Hb_MCH_Ratio' in df_noise.columns:
    df_noise.drop(columns=['Hb_MCH_Ratio', 'Hb_MCHC_Ratio', 'Hb_Gender_Interaction'], inplace=True, errors='ignore')

# Add new derived features without Hb
df_noise['RBC_MCV_Ratio'] = df_noise['RBC'] / df_noise['MCV']
df_noise['RDW_MCV_Ratio'] = df_noise['RDW'] / df_noise['MCV']
df_noise['MCH_MCHC_Ratio'] = df_noise['MCH'] / df_noise['MCHC']
df_noise['Age_RBC_Interaction'] = df_noise['Age'] * df_noise['RBC'] / 100
df_noise['RDW_Age_Interaction'] = df_noise['RDW'] * np.log(df_noise['Age'] + 1)

# Split into features and labels
# Explicitly exclude Hemoglobin from features
features_to_use = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'Age', 'Gender',
                   'RBC_MCV_Ratio', 'RDW_MCV_Ratio', 'MCH_MCHC_Ratio',
                   'Age_RBC_Interaction', 'RDW_Age_Interaction']
X = df_noise[features_to_use]

# Exclude TLC, PCV, and PLT parameters from the features
features_to_exclude = ['TLC', 'PCV', 'PLT']
for feature in features_to_exclude:
    if feature in X.columns:
        X = X.drop(columns=[feature])

print(f"Training model without excluded features: {features_to_exclude}")
print(f"Features used for training: {X.columns.tolist()}")

y = df_noise['Result']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Compare different models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Function to plot learning curves (to diagnose overfitting)
def plot_learning_curve(estimator, title, X, y, ax=None, ylim=None):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    
    ax.grid()
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    ax.legend(loc="best")
    ax.set_title(title)
    
    # Return the gap between train and test scores at the maximum number of samples
    gap = train_scores_mean[-1] - test_scores_mean[-1]
    return gap

print("\n--- Model Comparison with Cross-Validation ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
overfitting_gaps = {}

for i, (name, model) in enumerate(models.items()):
    # Perform 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    
    # Train on the main train set
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Calculate overfitting gap
    overfitting_gap = train_accuracy - test_accuracy
    overfitting_gaps[name] = overfitting_gap
    
    print(f"\nModel: {name}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Overfitting gap: {overfitting_gap:.4f}")
    print(f"Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Plot learning curve to diagnose overfitting
    gap = plot_learning_curve(
        model, f"Learning Curve - {name}", 
        X_scaled, y, ax=axes[i], 
        ylim=(0.5, 1.01)
    )
    
    # More detailed evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Anemia', 'Anemia'],
                yticklabels=['No Anemia', 'Anemia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'model/confusion_matrix_{name.replace(" ", "_").lower()}.png')

    # For logistic regression, extract and save coefficients
    if name == "Logistic Regression":
        coef = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0],
            'Abs_Coefficient': np.abs(model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nLogistic Regression Coefficients:")
        print(coef)
        
        # Plot coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef)
        plt.title('Feature Importance (Logistic Regression Coefficients)')
        plt.tight_layout()
        plt.savefig('model/logistic_regression_coefficients.png')
        
    # For Random Forest, extract and save feature importances
    if name == "Random Forest":
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(importances)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances)
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('model/random_forest_importance.png')

# Save the learning curves
plt.tight_layout()
plt.savefig('model/learning_curves.png')

# Create a bar plot showing overfitting gaps
plt.figure(figsize=(10, 6))
sns.barplot(x=list(overfitting_gaps.keys()), y=list(overfitting_gaps.values()))
plt.title('Overfitting Gap (Train Accuracy - Test Accuracy)')
plt.xlabel('Model')
plt.ylabel('Gap')
plt.axhline(0.05, linestyle='--', color='red', label='Threshold (0.05)')
plt.legend()
plt.tight_layout()
plt.savefig('model/overfitting_gap.png')

# If Random Forest shows significant overfitting, implement regularization
if overfitting_gaps.get("Random Forest", 0) > 0.05:
    print("\n--- Applying regularization to Random Forest model ---")
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    
    # Evaluate optimized model
    train_accuracy = best_rf.score(X_train, y_train)
    test_accuracy = best_rf.score(X_test, y_test)
    
    print(f"Optimized Random Forest:")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Overfitting gap: {train_accuracy - test_accuracy:.4f}")
    
    # Plot learning curve for optimized model
    plt.figure(figsize=(10, 6))
    plot_learning_curve(
        best_rf, "Learning Curve - Optimized Random Forest", 
        X_scaled, y, ylim=(0.5, 1.01)
    )
    plt.tight_layout()
    plt.savefig('model/optimized_rf_learning_curve.png')

# Create ROC curves for all models
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.savefig('model/roc_curves.png')

print("\n--- Overfitting Analysis Results ---")
print("Models with high overfitting risk:")
for name, gap in overfitting_gaps.items():
    risk_level = "HIGH" if gap > 0.1 else "MODERATE" if gap > 0.05 else "LOW"
    print(f"- {name}: {gap:.4f} ({risk_level} risk)")

print("\nOverfitting mitigation strategies implemented:")
print("1. Learning curves to visualize overfitting")
print("2. Regularization for models with high overfitting")
print("3. Cross-validation to get more stable performance estimates")
print("4. Hyperparameter tuning for Random Forest (if needed)")

print("\nAll visualizations saved to the 'model' directory")