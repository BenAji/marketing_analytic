# Customer Response Prediction App

This is a web application that allows users to upload a CSV file containing customer data and get predictions for customer responses using a trained machine learning model.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model file in the correct location:
- The model should be saved as `model/random_forest_tuned_model.pkl`
- You can save your trained model using:
```python
import pickle
with open('model/random_forest_tuned_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

## Running the App

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Using the App

1. Upload your CSV file using the file uploader
2. The file should contain the same features used to train the model
3. Click the "Make Predictions" button
4. View the predictions and download the results as a CSV file

## Input Data Format

Your CSV file should contain the following features:
- Customer demographic information
- Campaign and response history
- Interaction metrics
- Revenue information

The app will automatically handle:
- Missing value imputation
- Feature preprocessing
- Prediction generation

## Output

The app will add two new columns to your data:
- `predicted_response`: Binary prediction (0 or 1)
- `response_probability`: Probability of positive response 