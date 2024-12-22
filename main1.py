import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    Feature1 = np.random.uniform(1, 10, num_samples)  # Random numbers between 1 and 10
    Feature2 = Feature1 * 2 + np.random.normal(0, 1, num_samples)  # Add some noise
    Target = Feature1 * 3 + Feature2 * 1.5 + np.random.normal(0, 2, num_samples)  # Linear relationship + noise

    data = pd.DataFrame({
        'Feature1': Feature1,
        'Feature2': Feature2,
        'Target': Target
    })
    return data

# Step 2: Load Data into SQLite Database
def load_data_to_sqlite(data, db_name, table_name):
    # Connect to SQLite database (or create if not exists)
    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()

    # Create table dynamically based on DataFrame columns
    columns = ", ".join([f"{col} REAL" for col in data.columns])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    conn.commit()

    # Insert data into the table
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Step 3: Visualize Data
def visualize_data(data, x_col, y_col):
    fig, ax = plt.subplots()
    sns.lmplot(x=x_col, y=y_col, data=data, aspect=2, height=6)
    plt.title(f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    st.pyplot(fig)

# Step 4: Apply a Machine Learning Algorithm (Linear Regression)
def apply_linear_regression(data, feature_cols, target_col):
    X = data[feature_cols]
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# Streamlit App
def main():
    st.title("Data Warehouse and Machine Learning App")
    st.sidebar.title("Options")
    
    # User input for number of samples
    num_samples = st.sidebar.slider("Number of samples", min_value=50, max_value=1000, value=200, step=50)

    # Step 1: Generate Data
    st.header("1. Generate Synthetic Data")
    data = generate_synthetic_data(num_samples)
    st.write("Generated Data (First 5 rows):")
    st.write(data.head())

    # Step 2: Load Data into SQLite Database
    db_name = "warehouse"
    table_name = "synthetic_data"
    load_data_to_sqlite(data, db_name, table_name)
    st.success(f"Data loaded into SQLite database '{db_name}.db' in table '{table_name}'.")

    # Step 3: Visualize Data
    st.header("2. Visualize Data")
    x_col = st.selectbox("Select X-axis column:", data.columns)
    y_col = st.selectbox("Select Y-axis column:", data.columns)
    if st.button("Generate Plot"):
        visualize_data(data, x_col, y_col)

    # Step 4: Apply Machine Learning
    st.header("3. Apply Machine Learning")
    feature_cols = st.multiselect("Select feature columns:", data.columns, default=['Feature1', 'Feature2'])
    target_col = st.selectbox("Select target column:", data.columns)
    if st.button("Train Model"):
        model, mse, r2 = apply_linear_regression(data, feature_cols, target_col)
        st.write("Model Evaluation:")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

# Run the app
if __name__ == "__main__":
    main()
