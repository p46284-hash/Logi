import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Make sure 'Logi.sav' is in the same directory as this app.py file
try:
    model = joblib.load('Logi.sav')
except FileNotFoundError:
    st.error("Error: 'Logi.sav' not found. Please ensure the model file is in the same directory as app.py")
    st.stop()

st.title('Delivery Delay Prediction App')
st.write('Enter the feature values to predict if there will be a delivery delay.')

# Define input fields for each feature
# These correspond to the columns in your 'x' DataFrame:
# Delivery_Distance, Traffic_Congestion, Weather_Condition, Delivery_Slot, Driver_Experience, Num_Stops,
# Vehicle_Age, Road_Condition_Score, Package_Weight, Fuel_Efficiency, Warehouse_Processing_Time

Delivery_Distance = st.slider('Delivery Distance', 0.0, 100.0, 20.0)
Traffic_Congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
Weather_Condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
Delivery_Slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
Driver_Experience = st.slider('Driver Experience (Years)', 0, 20, 10)
Num_Stops = st.slider('Number of Stops', 0, 10, 5)
Vehicle_Age = st.slider('Vehicle Age (Years)', 0, 10, 5)
Road_Condition_Score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
Package_Weight = st.slider('Package Weight', 0.0, 50.0, 25.0)
Fuel_Efficiency = st.slider('Fuel Efficiency', 0.0, 30.0, 15.0)
Warehouse_Processing_Time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[Delivery_Distance, Traffic_Congestion, Weather_Condition,
                            Delivery_Slot, Driver_Experience, Num_Stops, Vehicle_Age,
                            Road_Condition_Score, Package_Weight, Fuel_Efficiency,
                            Warehouse_Processing_Time]],
                          columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot',
                                   'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score',
                                   'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'])

# Predict button
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write('---')
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Delivery Delay Predicted!')
    else:
        st.success('No Delivery Delay Predicted!')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")

st.write('---')
st.write('This app uses a Logistic Regression model to predict delivery delays.')
