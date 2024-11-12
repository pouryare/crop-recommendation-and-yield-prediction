"""
Streamlit Application for Crop Recommendation and Yield Prediction
===============================================================

This module implements a Streamlit web application for the crop recommendation
and yield prediction system. It provides an interactive interface for users
to input data and receive predictions.

Author: Pourya
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from typing import Tuple, Any
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="Crop Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths for models
MODEL_DIR = os.path.join(os.getcwd(), "models")
RECOMMENDATION_MODEL_PATH = os.path.join(MODEL_DIR, "recommendation.keras")
RECOMMENDATION_ENCODER_PATH = os.path.join(MODEL_DIR, "recommendation_encoder.joblib")
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, "yield_model.joblib")
YIELD_ENCODER_PATH = os.path.join(MODEL_DIR, "yield_encoder.joblib")

# Cache model loading
@st.cache_resource
def load_models() -> Tuple[Any, Any, Any, Any]:
    """
    Load all required models and encoders with caching.
    
    Returns:
        Tuple of (recommendation_model, yield_model, recommendation_encoder, yield_encoder)
    """
    try:
        recom_model = tf.keras.models.load_model(RECOMMENDATION_MODEL_PATH)
        yield_model = joblib.load(YIELD_MODEL_PATH)
        recom_encoder = joblib.load(RECOMMENDATION_ENCODER_PATH)
        yield_encoder = joblib.load(YIELD_ENCODER_PATH)
        return recom_model, yield_model, recom_encoder, yield_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

def create_recommendation_plot(input_data: dict) -> go.Figure:
    """
    Create a radar plot for input soil conditions.
    
    Args:
        input_data: Dictionary of input values
        
    Returns:
        Plotly figure object
    """
    categories = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    values = [
        input_data['nitrogen'],
        input_data['phosphorus'],
        input_data['potassium'],
        input_data['temperature'],
        input_data['humidity'],
        input_data['ph'],
        input_data['rainfall']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Soil Conditions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)]
            )),
        showlegend=False
    )
    
    return fig

def footer():
    """
    Display footer with attribution.
    """
    footer_html = """
    <div style="
        width: 100%;
        padding: 20px 0;
        text-align: center;
        ">
        <p style="
            margin: 0;
            font-size: 14px;
            color: #666;
            ">Made with ❤️ by Pourya</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit application."""
    
    # Load models
    with st.spinner('Loading models...'):
        recom_model, yield_model, recom_encoder, yield_encoder = load_models()
    
    st.title('Crop Analysis System')
    st.divider()
    
    # Create tabs
    tab1, tab2 = st.tabs(['Crop Recommendation', 'Yield Prediction'])
    
    # Crop Recommendation Tab
    with tab1:
        st.header('Crop Recommendation')
        st.write('Enter soil conditions and environmental factors to get crop recommendations.')
        
        with st.form("recommendation_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nitrogen = st.number_input('Nitrogen (N)', 0, 140, 50)
                phosphorus = st.number_input('Phosphorus (P)', 0, 145, 50)
                potassium = st.number_input('Potassium (K)', 0, 205, 50)
            
            with col2:
                temperature = st.number_input('Temperature (°C)', 0.0, 50.0, 25.0)
                humidity = st.number_input('Humidity (%)', 0.0, 100.0, 71.5)
            
            with col3:
                ph = st.number_input('pH', 0.0, 14.0, 6.5)
                rainfall = st.number_input('Rainfall (mm)', 0.0, 300.0, 103.0)
            
            submit_recommendation = st.form_submit_button('Get Recommendation')
        
        if submit_recommendation:
            with st.spinner('Analyzing soil conditions...'):
                # Prepare input data
                input_data = {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }
                
                # Create input array
                X = np.array([[
                    nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall
                ]])
                
                # Get prediction
                prediction = recom_model.predict(X)
                predicted_crop = recom_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                confidence = float(np.max(prediction[0]) * 100)
                
                # Display results
                st.subheader('Recommendation Results')
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Recommended Crop', predicted_crop)
                with col2:
                    st.metric('Confidence', f"{confidence:.1f}%")
                
                # Display radar plot
                st.subheader('Soil Condition Analysis')
                fig = create_recommendation_plot(input_data)
                st.plotly_chart(fig, use_container_width=True)
    
    # Yield Prediction Tab
    with tab2:
        st.header('Yield Prediction')
        st.write('Predict crop yield based on environmental conditions.')
        
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                crop = st.selectbox(
                    'Select Crop',
                    options=yield_encoder.classes_
                )
                year = st.number_input('Year', 1990, 2024, 2024)
            
            with col2:
                rainfall = st.number_input(
                    'Average Rainfall (mm/year)',
                    0.0, 5000.0, 1083.0,
                    key='yield_rainfall'
                )
                pesticides = st.number_input(
                    'Pesticides (tonnes)',
                    0.0, 100000.0, 45620.0
                )
                avg_temp = st.number_input(
                    'Average Temperature (°C)',
                    0.0, 50.0, 25.0,
                    key='yield_temp'
                )
            
            submit_yield = st.form_submit_button('Predict Yield')
        
        if submit_yield:
            with st.spinner('Calculating yield prediction...'):
                # Prepare input data
                crop_encoded = yield_encoder.transform([crop])[0]
                X = pd.DataFrame([[crop_encoded, year, rainfall, pesticides, avg_temp]],
                               columns=['Item', 'Year', 'average_rain_fall_mm_per_year',
                                      'pesticides_tonnes', 'avg_temp'])
                
                # Get prediction
                predicted_yield = yield_model.predict(X)[0]
                
                # Display results
                st.subheader('Yield Prediction Results')
                
                # Display metric
                st.metric(
                    'Predicted Yield',
                    f"{predicted_yield:.2f} hg/ha",
                    delta=None
                )
                
                # Create and display yields comparison chart
                reference_years = range(year-5, year+1)
                reference_predictions = []
                
                for ref_year in reference_years:
                    X_ref = pd.DataFrame(
                        [[crop_encoded, ref_year, rainfall, pesticides, avg_temp]],
                        columns=X.columns
                    )
                    ref_pred = yield_model.predict(X_ref)[0]
                    reference_predictions.append(ref_pred)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(reference_years),
                    y=reference_predictions,
                    mode='lines+markers',
                    name='Predicted Yield'
                ))
                fig.update_layout(
                    title=f'Yield Trends for {crop}',
                    xaxis_title='Year',
                    yaxis_title='Yield (hg/ha)',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
    footer()