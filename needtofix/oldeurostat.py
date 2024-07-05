import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Eurostat API base URL
BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

def fetch_eurostat_data(dataset_code, params):
    url = f"{BASE_URL}{dataset_code}"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

def process_data(data):
    if not data or 'value' not in data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data['value'].items(), columns=['id', 'value'])
    
    # Split the id into separate columns
    id_cols = data['dimension']['id']
    df[id_cols] = df['id'].str.split(',', expand=True)
    
    # Replace codes with labels
    for col in id_cols:
        df[col] = df[col].map(data['dimension'][col]['category']['label'])
    
    return df

st.title("Eurostat Data Explorer")

# Sidebar for user input
st.sidebar.header("Data Selection")
dataset_code = st.sidebar.text_input("Enter Eurostat dataset code", "nama_10_gdp")

# Fetch and display the data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        data = fetch_eurostat_data(dataset_code, params={'format': 'JSON'})
        if data:
            df = process_data(data)
            st.write(df)

            # Simple visualization
            if not df.empty:
                st.subheader("Data Visualization")
                x_axis = st.selectbox("Select X-axis", df.columns)
                y_axis = st.selectbox("Select Y-axis", df.columns)
                
                fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=df.columns)
                st.plotly_chart(fig)

# Add information about the application
st.sidebar.markdown("---")
st.sidebar.info("This application uses the Eurostat API to fetch and visualize statistical data.")
st.sidebar.info("Enter a dataset code in the input box above and click 'Fetch Data' to start.")

# For debugging: Print the first 500 characters of the raw API response
if st.sidebar.button("Debug: Show Raw API Response"):
    debug_data = fetch_eurostat_data(dataset_code, params={'format': 'JSON'})
    if debug_data:
        st.code(str(debug_data)[:500], language='json')