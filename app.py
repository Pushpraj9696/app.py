import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Prediction (IMDb India)")
st.write("This app predicts the IMDb rating of Indian movies using Machine Learning.")

# Load and train model (cached so it doesn't run on every click)
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
    df = df.dropna(subset=['Rating'])
    
    # Data Cleaning
    df['Year'] = df['Year'].str.extract(r'(\d+)').astype(float)
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
    
    # Fill Missing values
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())
    df['Votes'] = df['Votes'].fillna(df['Votes'].median())
    
    categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    df[categorical_features] = df[categorical_features].fillna('Unknown')
    
    X = df[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
    y = df['Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)) # Reduced estimators for faster web load
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Calculate performance
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get unique lists for dropdowns
    unique_genres = sorted(df['Genre'].unique().tolist())
    unique_directors = sorted(df['Director'].unique().tolist())
    unique_actors = sorted(df['Actor 1'].dropna().unique().tolist())
    
    return model_pipeline, mae, unique_genres, unique_directors, unique_actors

with st.spinner("Loading dataset and training model... Please wait..."):
    model, model_mae, genres, directors, actors = train_model()

st.sidebar.success(f"🤖 Model Loaded! (MAE: {model_mae:.2f})")

# User Inputs
st.header("🎯 Enter Movie Details")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year of Release", min_value=1900, max_value=2026, value=2024)
    duration = st.number_input("Duration (in minutes)", min_value=30, max_value=300, value=130)
    votes = st.number_input("Expected/Current Votes", min_value=0, max_value=1000000, value=5000)

with col2:
    genre = st.selectbox("Genre", genres)
    director = st.selectbox("Director", directors)
    actor1 = st.selectbox("Lead Actor 1", actors)
    actor2 = st.selectbox("Lead Actor 2", actors)
    actor3 = st.selectbox("Lead Actor 3", actors)

# Predict Button
if st.button("🚀 Predict IMDb Rating", type="primary"):
    input_data = pd.DataFrame([{
        'Year': float(year),
        'Duration': float(duration),
        'Votes': float(votes),
        'Genre': genre,
        'Director': director,
        'Actor 1': actor1,
        'Actor 2': actor2,
        'Actor 3': actor3
    }])
    
    prediction = model.predict(input_data)[0]
    st.balloons()
    st.metric(label="Predicted IMDb Rating", value=f"⭐ {prediction:.1f} / 10")
  
