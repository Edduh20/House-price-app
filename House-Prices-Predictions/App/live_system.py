import pandas as pd
import sklearn
import streamlit as st
from joblib import load

st.title("House Price Prediction System")

neighbourhood_dict = {
"Bloomington Heights": "Blmngtn",
    "Bluestem": "Blueste",
    "Briardale": "BrDale",
    "Brookside": "BrkSide",
    "Clear Creek": "ClearCr",
    "College Creek": "CollgCr",
    "Crawford": "Crawfor",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "Iowa DOT and Rail Road": "IDOTRR",
    "Meadow Village": "MeadowV",
    "Mitchell": "Mitchel",
    "North Ames": "Names",
    "Northridge": "NoRidge",
    "Northpark Villa": "NPkVill",
    "Northridge Heights": "NridgHt",
    "Northwest Ames": "NWAmes",
    "Old Town": "OldTown",
    "South & West of Iowa State University": "SWISU",
    "Sawyer": "Sawyer",
    "Sawyer West": "SawyerW",
    "Somerset": "Somerst",
    "Stone Brook": "StoneBr",
    "Timberland": "Timber",
    "Veenker": "Veenker"
}

house_style_dict = {
	"One story": "1Story",
    "1.5Fin	One and one-half story - 2nd level finished": "1.5Fin",
    "One and one-half story - 2nd level unfinished": "1.5Unf",
    "2Story": "Two story",
    "Two and one-half story - 2nd level finished": "2.5Fin",
    "Two and one-half story - 2nd level unfinished": "2.5Unf",
    "Split Foyer": "SFoyer",
    "Split Level": "SLvl"
}

st.write("Choose your prefered House features for Sale price prediction")
neighbourhood  = st.selectbox("Select your preferred neighbourhood",
                              neighbourhood_dict.keys())

house_style  = st.selectbox("Select your preferred housestyle",
                                  house_style_dict.keys())

overall_score = st.slider(label ="Choose your overall score of the house\n"
                          "\nOverall Condition and Quality(1 = Very Poor, 10 = Very Excellent)",
                          min_value=1.0,
                          max_value=10.0,
                          step=0.5)

Renovation_age = st.number_input("Input Rennovation age", min_value=1, step=1)
st.write("The Renovation age is:", Renovation_age)

Tot_bathrooms = st.number_input("Input Total bathrooms of the house", min_value=1.0, step=0.5)
st.write("The total number of bathrooms are:", Tot_bathrooms)

Tot_bedrooms = st.number_input("Input Total bedrooms of the house", min_value=1, step=1)
st.write("The total number of bedrooms are:", Tot_bedrooms)

Tot_rooms = st.number_input("Input Total rooms of the house", min_value=1, step=1)
st.write("The total number of rooms in the house are:", Tot_rooms)

# Accessing the values of the dictionary
neighbourhood_values = neighbourhood_dict[neighbourhood]
house_style_values = house_style_dict[house_style]

model = load("House-Prices-Predictions/Models/random_forest.joblib")
trained_columns = load("House-Prices-Predictions/Models/model_columns.pkl")


df = pd.DataFrame({
    'Neighborhood': [neighbourhood_values],
    'HouseStyle': [house_style_values],
    'Overall_score': [overall_score],
    'Remod_age': [Renovation_age],
    'Tot_bathrooms': [Tot_bathrooms],
    'Tot_bedrooms': [Tot_bedrooms],
    'Tot_rooms': [Tot_rooms]
})

df = pd.get_dummies(df, columns=['Neighborhood', 'HouseStyle'], dtype = int, drop_first=True)
df = df.reindex(columns=trained_columns, fill_value=0)


if st.button("Predict Price"):
    prediction = model.predict(df)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
