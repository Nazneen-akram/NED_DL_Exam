import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the pre-trained model
# model = load_model('bitcoin_price_prediction_model.h5')

# Load the Bitcoin dataset
data = pd.DataFrame({
    'Open': [357, 363, 365.7, 361, 356.45, 355.51, 355.01],
    'Max': [364.98, 369.88, 365.9, 362.48, 359.9, 357.5, 361.99],
    'Min': [356, 363, 360, 356.1, 351.29, 351.9, 353.88],
    'Close': [362.06, 367.85, 363.37, 356.84, 352.39, 352.91, 355],
    'Volume': [627500, 1232500, 1228400, 1509000, 1524900, 1413300, 2693100],
    'Margin': [720.02, 731.865, 727.485, 718.21, 710.015, 708.91, 712.94],
    'Date': pd.to_datetime(['1/1/2015', '1/2/2015', '1/5/2015', '1/6/2015', '1/7/2015', '1/8/2015', '1/9/2015'])
})

# Convert the date to a numerical value
data['Date'] = (data['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Function to preprocess user input for prediction
def preprocess_input(user_input, scaler):
    # Assume user input is in the same format as the training data
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    user_input_reshaped = np.reshape(user_input_scaled, (1, user_input_scaled.shape[0], 1))
    return user_input_reshaped

# Function to make predictions
def make_prediction(model, user_input):
    prediction = model.predict(user_input)
    return prediction

# Streamlit UI
st.title('Bitcoin Price Prediction App')

# User input section
st.sidebar.header('User Input')
user_input = {}

# Example input fields (replace with actual features in your dataset)
user_input['Open'] = st.sidebar.slider('Open', min_value=0.0, max_value=500.0, value=250.0)
user_input['Max'] = st.sidebar.slider('Max', min_value=0.0, max_value=500.0, value=250.0)
user_input['Min'] = st.sidebar.slider('Min', min_value=0.0, max_value=500.0, value=250.0)
user_input['Close'] = st.sidebar.slider('Close', min_value=0.0, max_value=500.0, value=250.0)
user_input['Volume'] = st.sidebar.slider('Volume', min_value=0, max_value=3000000, value=1500000)
user_input['Margin'] = st.sidebar.slider('Margin', min_value=0.0, max_value=1000.0, value=500.0)

def preprocess_input(user_input, scaler):
    # Assume user input is in the same format as the training data
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    user_input_reshaped = np.reshape(user_input_scaled, (1, 1, user_input_scaled.shape[1]))
    return user_input_reshaped

# Make predictions
# prediction = make_prediction(model, user_input_scaled)

# Output section
st.header('Prediction')
# st.write(f'The predicted Bitcoin price is: {prediction[0][0]}')

# Display your name and CNIC
st.sidebar.header('Output Details')
st.sidebar.write('Name: Nazneen AKram')
st.sidebar.write('CNIC: 4210145447712')
