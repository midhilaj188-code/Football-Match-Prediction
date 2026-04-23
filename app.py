import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("⚽ Football Prediction App")

# 👉 Integer inputs better
f1 = st.number_input("Halftime Home Goals", min_value=0, step=1)
f2 = st.number_input("Halftime Away Goals", min_value=0, step=1)
f3 = st.number_input("Total Goals", min_value=0, step=1)
f4 = st.number_input("Goal Difference", step=1)

if st.button("Predict"):

    # Validation
    if abs(f4) > f3:
        st.error("Invalid input! Goal difference too large")

    elif (f3 + f4) % 2 != 0:
        st.error("Invalid combination of Total Goals & Goal Difference")

    else:
        data = np.array([[f1, f2, f3, f4]])
        pred = model.predict(data)

        if pred[0] == 1:
            st.success("Home Team Wins")
        else:
            st.error("Home Team Loses")