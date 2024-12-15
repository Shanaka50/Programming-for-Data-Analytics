import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer

# Load Data Function
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    # Drop rows with missing PM2.5 values
    df.dropna(subset=['PM2.5'], inplace=True)
    
    # Pad month, day, and hour with leading zeros to ensure proper format
    df['year'] = df['year'].astype(str)
    df['month'] = df['month'].apply(lambda x: f"{int(x):02d}")
    df['day'] = df['day'].apply(lambda x: f"{int(x):02d}")
    df['hour'] = df['hour'].apply(lambda x: f"{int(x):02d}")
    
    # Combine columns to create a datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    
    # Drop original date-related columns
    df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)
    return df


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipelines
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean for numerical columns
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with mode for categorical columns
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    # Models
    models = {
        "Linear Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        "Support Vector Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR())
        ])
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"MSE": mse, "R2": r2}

    return results


# Streamlit Application
st.title("Beijing Multi-Site Air Quality Analysis")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Data Overview", "EDA", "Feature Engineering", "Modeling & Prediction", "Prediction"])

if page == "Data Overview":
    st.header("Data Overview")
    uploaded_file = st.file_uploader("Upload a Dataset", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)
        st.write("### Raw Data Sample")
        st.dataframe(data.head())
        st.write("### Data Summary")
        st.write(data.describe())
        st.write("### Missing Values")
        st.write(data.isnull().sum())

if page == "EDA":
    st.header("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload a Dataset", type=["csv"])

    if uploaded_file:
        data = preprocess_data(load_data(uploaded_file))
        st.write("### PM2.5 Distribution")
        plt.figure(figsize=(8, 5))
        sns.histplot(data['PM2.5'], bins=50, kde=True, color='skyblue')
        st.pyplot(plt)

        st.write("### Correlation Heatmap (Numeric Features)")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)


if page == "Feature Engineering":
    st.header("Feature Engineering")
    uploaded_file = st.file_uploader("Upload a Dataset", type=["csv"])

    if uploaded_file:
        data = preprocess_data(load_data(uploaded_file))
        st.write("### Original Data")
        st.dataframe(data.head())

        st.write("### Add Pollution Index Feature")
        data['pollution_index'] = data[['PM2.5', 'PM10', 'NO2']].sum(axis=1)
        st.write("### Updated Data with Pollution Index")
        st.dataframe(data.head())

        st.write("### PM2.5 vs Pollution Index")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=data['pollution_index'], y=data['PM2.5'], color='red')
        plt.title("PM2.5 vs Pollution Index")
        st.pyplot(plt)

if page == "Modeling & Prediction":
    st.header("Modeling & Prediction")
    uploaded_file = st.file_uploader("Upload a Dataset", type=["csv"])

    if uploaded_file:
        data = preprocess_data(load_data(uploaded_file))
        X = data.drop(columns=['PM2.5', 'datetime'])
        y = data['PM2.5']

        st.write("### Model Training")
        results = train_models(X, y)
        st.write("### Model Performance")
        st.table(pd.DataFrame(results).T)

        st.write("### Performance Comparison")
        df_results = pd.DataFrame(results).T
        plt.figure(figsize=(10, 6))
        df_results['R2'].plot(kind='bar', color='lightgreen', title='R2 Score Comparison')
        st.pyplot(plt)

if page == "Prediction":
    st.header("Prediction on New Data")
    uploaded_file = st.file_uploader("Upload Preprocessed Dataset", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)
        st.write("### Input Data")
        st.dataframe(data.head())

        # Allow user input for prediction
        st.write("### Enter Feature Values")
        input_data = {}
        for col in data.columns.drop(['PM2.5', 'datetime']):
            input_data[col] = st.number_input(f"Enter {col}", float(data[col].min()), float(data[col].max()))
        input_df = pd.DataFrame([input_data])

        # Train a Random Forest for prediction
        X = data.drop(columns=['PM2.5', 'datetime'])
        y = data['PM2.5']
        rf = RandomForestRegressor()
        rf.fit(X, y)
        prediction = rf.predict(input_df)

        st.write("### Prediction for PM2.5")
        st.success(f"Predicted PM2.5 Value: {prediction[0]:.2f}")
