
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer as ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(FILE_DIR, "resources")

MODEL_PATH = os.path.join(dir_of_interest, 'data', "model.pkl")
DF_PATH = os.path.join(dir_of_interest, 'data', "df.pkl")

model = pickle.load(open(MODEL_PATH, 'rb'))
df = pickle.load(open(DF_PATH, 'rb'))

df = pd.DataFrame(df)

def main():
    # ML Section
    features = ["Brand", "Processor", "RAM", "OS", "Storage"]
    f = df[["Brand", "Processor", "RAM", "OS", "Storage"]]
    y = np.log(df['MRP'])
    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=47)
    step1 = ct(transformers=[
        ('encoder',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,4])
    ],remainder='passthrough')

    step2 = RandomForestRegressor(n_estimators=100,
                                random_state=3,
                                max_samples=0.5,
                                max_features=0.75,
                                max_depth=15)

    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe.fit(X_train,y_train)

    st.title('Laptop Price Prediction: :computer:')

    # input variables
    brand = st.selectbox("Select any Brand:- ", df["Brand"].unique())
    processor = st.selectbox("Select any Processor:- ", df["Processor"].unique())
    ram = st.selectbox("Select any RAM:- ", df["RAM"].unique())
    os = st.selectbox("Select any Operating System:- ", df["OS"].unique())
    Storage = st.selectbox("Select any Storage:- ", df["Storage"].unique())

    # Model Prediction
    if st.button('Predict'):
        st.snow()
        query = np.array([brand, processor, ram, os, Storage])
        query = query.reshape(1, -1)
        p = pipe.predict(query)[0]
        result = np.exp(p)
        st.success("₹{}".format(round(result, 2)))
    
if __name__=='__main__':
    st.balloons()
    main()