import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.impute import SimpleImputer

# Create model directory if it doesn't exist
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Define model path consistently
MODEL_PATH = 'model/random_forest_tuned_model.pkl'

# Constants for income normalization (from training data)
INCOME_MIN = 20000
INCOME_MAX = 250000
INCOME_MEAN = 74943.27  # Using middle point as approximation, you should replace with actual mean from training
INCOME_STD = 52138.96   # Using approximate std, you should replace with actual std from training

# Add error handling for model loading
try:
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.error("Please make sure you have trained and saved the model first.")
        st.info("The model should be saved using:")
        st.code("""
# Save the model correctly
with open('model/random_forest_tuned_model.pkl', 'wb') as file:
    pickle.dump(model, file)
            """)
        st.stop()
    
    # Try to load the model
    with open(MODEL_PATH, 'rb') as file:
        try:
            model = pickle.load(file)
            # Get feature names from the model if available
            if hasattr(model, 'feature_names_in_'):
                MODEL_FEATURES = list(model.feature_names_in_)
            else:
                st.error("Model doesn't have feature names stored. Please retrain the model with feature names.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")
            st.error("Please make sure the model was saved correctly using pickle.dump() in binary mode ('wb')")
            st.stop()

except Exception as e:
    st.error(f"An error occurred while setting up the application: {str(e)}")
    st.stop()

# Set page title and description
st.title('Customer Response Prediction App')
st.write("""
### Upload your CSV file for prediction
Please ensure your CSV file contains the necessary features for prediction.
""")

# Display expected features
st.write("### Expected Features:")
st.write("Your CSV file should contain these features:")
st.write(MODEL_FEATURES)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def preprocess_data(df):
    """
    Preprocess the input dataframe to match the model's expected features.
    """
    # Check which expected features are missing
    missing_features = set(MODEL_FEATURES) - set(df.columns)
    if missing_features:
        st.warning(f"Missing features in uploaded file: {missing_features}")
        # Add missing features with default value of 0
        for feature in missing_features:
            df[feature] = 0
    
    # Check for extra features
    extra_features = set(df.columns) - set(MODEL_FEATURES)
    if extra_features:
        st.info(f"Removing extra features not used by the model: {extra_features}")
    
    # Select only the features the model expects, in the correct order
    features = df[MODEL_FEATURES]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(features)
    
    return X_imputed

def main():
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview (Before Preprocessing):")
            st.write(df.head())
            
            if st.button('Make Predictions'):
                # Preprocess the data
                X = preprocess_data(df)
                
                # Make predictions
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
                
                # Add predictions to the dataframe
                df['predicted_response'] = predictions
                df['response_probability'] = probabilities
                
                st.write("### Predictions Preview:")
                st.write(df.head())
                
                # Create download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Show prediction statistics
                st.write("### Prediction Statistics:")
                st.write(f"Total predictions: {len(predictions)}")
                st.write(f"Predicted responses (1): {sum(predictions)}")
                st.write(f"Predicted non-responses (0): {len(predictions) - sum(predictions)}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure your CSV file has the correct format.")
            st.write("Error details:", str(e))

if __name__ == '__main__':
    main() 