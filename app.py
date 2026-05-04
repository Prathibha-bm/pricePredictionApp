import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model_knn.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title(" House Price Prediction System")

st.write("Enter house details to predict the price")

square_footage = st.number_input("Square Footage", min_value=0.0)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=0)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=0)
year_built = st.number_input("Year Built", min_value=1800)
lot_size = st.number_input("Lot Size", min_value=0.0)
garage_size = st.number_input("Garage Size", min_value=0)


neighborhood_quality = st.selectbox(
    "Neighborhood Quality",
    [0, 1, 2, 3, 4]
)


if st.button("Predict Price"):

    input_data = np.array([[square_footage,
                            num_bedrooms,
                            num_bathrooms,
                            year_built,
                            lot_size,
                            garage_size,
                            neighborhood_quality]])

    input_data = np.log1p(input_data)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    final_price = np.expm1(prediction[0])

    st.success(f"Estimated House Price: $ {final_price:,.2f}")