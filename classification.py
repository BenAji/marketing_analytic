import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#import xgboost as XGBClassifier
import os
from datetime import datetime
import pickle


# Create output directories
output_dir = 'data'
#plots_dir = 'plots'
models_dir = 'model'
#os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("Starting Classification Modeling for Response Prediction...")

# Load the engineered features dataset
print("Loading engineered features...")
modeling_df = pd.read_csv(f"{output_dir}/modeling_features.csv", nrows=20000)

# 1. Prepare data for modeling
print("\n--- Preparing Data for Modeling ---")

# Define features and target
X = modeling_df.drop(['customer_id', 'has_responded', 'total_revenue'], axis=1)
y = modeling_df['has_responded']

# Convert features to numpy array to remove feature names
# X = X.to_numpy()

# Check for missing values
missing_values = np.isnan(X).sum(axis=0)
missing_cols = np.where(missing_values > 0)[0]

# Class distribution
# class_counts = y.value_counts()
# class_percentages = class_counts / len(y) * 100

# Imbalance ratio
# imbalance_ratio = class_counts[0] / class_counts[1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Handle missing values with imputation
print("\n--- Handling Missing Values with Imputation ---")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

#  Convert test data to numpy array as well to ensure consistency
print("\n--- convert x_test to numpy ---")
if isinstance(X_test_imputed, pd.DataFrame):
    X_test_imputed = X_test_imputed.to_numpy()

# 3. Handle class imbalance with SMOTE
print("\n--- Handling Class Imbalance with SMOTE ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imputed, y_train)

#     # Class distribution after SMOTE
# smote_class_counts = pd.Series(y_train_smote).value_counts()
# smote_class_percentages = smote_class_counts / len(y_train_smote) * 100

# 4. Define evaluation metrics
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    

    # Return metrics
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        # 'confusion_matrix': cm,
        'model': model
    }


# 5. Train and evaluate multiple models
print("\n--- Training and Evaluating Multiple Models ---")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    #'XGBoost': XGBClassifier.XGBClassifier(scale_pos_weight=imbalance_ratio, random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}


# Train and evaluate each model
results = []
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")

    # Store feature names before converting to numpy array
    feature_names = list(X_train.columns)

    result = evaluate_model(model, X_train_smote, y_train_smote, X_test_imputed, y_test, model_name)
    results.append(result)
    
        # Set feature names for the model
    if isinstance(model, RandomForestClassifier):
        model.feature_names_in_ = feature_names
        
    # Save the model using pickle
    # For saving individual models in the loop
    with open(f"{models_dir}/{model_name.lower().replace(' ', '_')}_model.pkl", 'wb') as file:
        pickle.dump(model, file)
# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_score', ascending=False)


# 6. Hyperparameter tuning for the best model
print("\n--- Performing Hyperparameter Tuning ---")

# Identify the best model based on F1-score
best_model_name = results_df.iloc[0]['model_name']
print(f"Best model based on F1-score: {best_model_name}")

# Define hyperparameter grids for each model type
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    #},
    #'XGBoost': {
    #    'n_estimators': [100, 200, 300],
    #    'learning_rate': [0.01, 0.1, 0.2],
    #    'max_depth': [3, 5, 7],
    #    'subsample': [0.8, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }
}



# Get the parameter grid for the best model
best_param_grid = param_grids[best_model_name]

# Create a new instance of the best model
if best_model_name == 'Logistic Regression':
    best_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
elif best_model_name == 'Random Forest':
    best_model = RandomForestClassifier(class_weight='balanced', random_state=42)
elif best_model_name == 'Gradient Boosting':
    best_model = GradientBoostingClassifier(random_state=42)
#elif best_model_name == 'XGBoost':
#    best_model = XGBClassifier.XGBClassifier(scale_pos_weight=imbalance_ratio, random_state=42)
elif best_model_name == 'SVM':
    best_model = SVC(class_weight='balanced', probability=True, random_state=42)
elif best_model_name == 'K-Nearest Neighbors':
    best_model = KNeighborsClassifier()
elif best_model_name == 'Naive Bayes':
    best_model = GaussianNB()

# Perform grid search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=best_param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

# Get the best parameters and model
best_params = grid_search.best_params_
best_tuned_model = grid_search.best_estimator_

# Evaluate the tuned model
tuned_result = evaluate_model(best_tuned_model, X_train_smote, y_train_smote, X_test_imputed, y_test, f"{best_model_name} (Tuned)")


# Set feature names for the tuned model
if isinstance(best_tuned_model, RandomForestClassifier):
    best_tuned_model.feature_names_in_ = feature_names

# For saving the tuned model
with open(f"{models_dir}/{best_model_name.lower().replace(' ', '_')}_tuned_model.pkl", 'wb') as file:
    pickle.dump(best_tuned_model, file)