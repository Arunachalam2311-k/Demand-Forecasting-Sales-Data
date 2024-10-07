# import librarys
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Set the page configuration to have a wide layout
st.set_page_config(layout='wide')

# Add a title to the Streamlit app
st.markdown("<h1 style='text-align: center; color: blue;'>Demand Forecasting Sales Data Dashboard</h1>", unsafe_allow_html=True)

# Create three tabs: HOME, PREDICTION, CONCLUSION
tab1, tab2, tab3 = st.tabs(['HOME', 'PREDICTION', 'CONCLUSION'])

# HOME tab content
with tab1:
    st.write(""" # 
    The company has accumulated transactional data from December 2021 to December 2023, 
    consisting of stock code, transaction dates, and quantities sold, etc. 
    The company requires a system to forecast the demand for the next 15 weeks 
    for its top 10 best-selling products. The goal is to estimate future demand 
    accurately to maintain optimal stock levels, ensuring that the supply chain 
    remains efficient and meets customer demands. The solution should leverage 
    the historical data to predict future demand trends and support inventory 
     management decisions.
    """)

# Load the model, scaler, and label encoders
model_path = r'D:\model\xgboost_regressor.pkl'
scaler_path = r'D:\model\xg_scaling.pkl' 
label_encoder_path = r'D:\model\encoded_data.pkl'  

# Load model
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error("Model file not found.")

# Load scaler
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
else:
    st.error("Scaler file not found.")

# Load label encoders
if os.path.exists(label_encoder_path):
    with open(label_encoder_path, 'rb') as file:
        label_encoders = pickle.load(file)
else:
    st.error("Label encoder file not found.")

# Function to transform categorical features using the loaded label encoders
def transform_with_fallback(label_encoder, value):
    """Transform value using the provided label encoder, returning -1 for unknown categories."""
    try:
        return label_encoder.transform([value])[0]
    except ValueError:
        return -1 

# PREDICTION tab content
with tab2:
    col1, col2 = st.columns(2, gap='large')

    # Predefined stock codes
    stock_codes = ["85123A", "22423", "85099B", "21523", "48173C", "48188", "21955", "20685", "21181", "21623"]

    # List of days, months, and years for selection
    days = list(range(1, 32))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    years = [2021, 2022, 2023]
    countries = ["United Kingdom", "France", "Australia", "Belgium", "Netherlands", "USA"]

    with col1:
        # Day, Month, and Year input for prediction
        selected_day = st.selectbox("Select a day", days)
        selected_month = st.selectbox("Select a month", months)
        selected_year = st.selectbox("Select a year", years)

    with col2:
        # Stock code selection
        selected_stock_code = st.selectbox("Select a stock code", stock_codes)
        selected_country = st.selectbox("Select a country", countries)
        Quantity = st.number_input("Insert a Quantity number", step=1, min_value=0)
        Price = st.number_input("Insert a Price", step=1.00, min_value=0.0)  

    # Convert selected day, month, and year to a full date
    selected_date = pd.to_datetime(f'{selected_year}-{months.index(selected_month) + 1}-{selected_day}')


    # Prepare input data for prediction
    if st.button('Predict'):
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'Country': [selected_country],
            'StockCode': [selected_stock_code],
            'Quantity': [Quantity],
            'Price': [Price],
            'Day': [selected_day],
            'Month': [months.index(selected_month) + 1], 
            'Year': [selected_year]
        })

        # Transform categorical features using the loaded label encoders
        if 'Country' in label_encoders:
            input_data['Country'] = transform_with_fallback(label_encoders['Country'], input_data['Country'][0])
        if 'StockCode' in label_encoders:
            input_data['StockCode'] = transform_with_fallback(label_encoders['StockCode'], input_data['StockCode'][0])

        # Prepare the input features for the model
        features = input_data[['Country', 'StockCode', 'Quantity', 'Price', 'Day', 'Month', 'Year']].values
        
        # Apply scaling if scaler is loaded
        if scaler:
            features = scaler.transform(features)

        # Make predictions using the model
        if model:
            forecast_values = model.predict(features)
            predicted_value = forecast_values[0]

            # Display the forecasted value
            # st.success(f"Predicted demand for stock code {selected_stock_code} on {selected_date.date()} is {predicted_value}")

            # Generate the forecasted demand for the next 15 weeks
            forecasted_demand = []
            future_dates = []

            for i in range(1, 16):
                future_date = selected_date + pd.DateOffset(weeks=i)
                future_dates.append(future_date)

                # Prepare future input data
                input_data['Day'] = future_date.day
                input_data['Month'] = future_date.month
                input_data['Year'] = future_date.year

                # Transform categorical features again
                input_data['Country'] = transform_with_fallback(label_encoders['Country'], selected_country)
                input_data['StockCode'] = transform_with_fallback(label_encoders['StockCode'], selected_stock_code)

                features = input_data[['Country', 'StockCode', 'Quantity', 'Price', 'Day', 'Month', 'Year']].values

                if scaler:
                    features = scaler.transform(features)

                forecasted_value = model.predict(features)
                forecasted_demand.append(forecasted_value[0])

            # Create DataFrame for forecasted demand for plotting
            demand_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Demand': forecasted_demand
            })

            # Historical demand data
            historical_demand = [10, 12, 15, 20, 18, 22, 24, 30, 28, 25, 20, 18, 15, 12, 10]  
            historical_dates = pd.date_range(start=selected_date - pd.DateOffset(weeks=15), periods=15)  
            historical_df = pd.DataFrame({
                'Date': historical_dates,
                'Historical Demand': historical_demand
            })

            # Combine historical and forecasted data for plotting
            combined_df = pd.concat([historical_df.set_index('Date'), demand_df.set_index('Date')], axis=1)



            # Bar chart for forecasted demand
            st.write("### Bar Chart of Forecasted Demand for the Next 15 Weeks")
            st.bar_chart(demand_df.set_index('Date'))

            # Plotting the forecasted demand for the next 15 weeks
            st.write("### Combined Historical and Forecasted Demand")
            st.line_chart(demand_df)

            # Histogram of forecasted demand
            st.write("### Histogram of Forecasted Demand")
            st.bar_chart(demand_df['Forecasted Demand']) 

            # Area chart of forecasted demand
            st.write("### Area Chart of Forecasted Demand")
            st.area_chart(demand_df.set_index('Date'))

 

# CONCLUSION tab content
with tab3:
    st.write("""#
             
The Demand Forecasting Sales Data Dashboard has effectively integrated historical sales data
 to predict future demand trends for the company's top 10 selling products.
 By leveraging advanced forecasting techniques, the dashboard provides valuable insights that enable the
 company to optimize inventory management and ensure adequate stock levels.
 The visualization of demand forecasts for the next 15 weeks empowers stakeholders to make informed 
decisions regarding these top products, ultimately improving operational efficiency and enhancing customer satisfaction.
This project demonstrates the importance of data-driven strategies in navigating the complexities of demand management in today's
dynamic market environment.
 
    """)