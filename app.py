import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/heart.csv")

# Features & target
X = data[["age", "trestbps", "chol", "thalach"]]
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Train models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Accuracy
lr_acc = accuracy_score(y_test, lr.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# ---------------- UI ---------------- #

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details:")

age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)
hr = st.number_input("Heart Rate", 60, 200)

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"]
)

# Prediction
if st.button("Predict"):
    input_data = [[age, bp, chol, hr]]

    if model_choice == "Logistic Regression":
        pred = lr.predict(input_data)
    elif model_choice == "Decision Tree":
        pred = dt.predict(input_data)
    else:
        pred = rf.predict(input_data)

    if pred[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")

# ---------------- Charts ---------------- #

# Show clean accuracy values first
st.subheader("📌 Model Accuracy Values")

st.write("Logistic Regression Accuracy:", round(lr_acc, 2))
st.write("Decision Tree Accuracy:", round(dt_acc, 2))
st.write("Random Forest Accuracy:", round(rf_acc, 2))


# ---------------- Charts ---------------- #

st.subheader("📊 Individual Model Performance")

# Logistic Regression Chart
fig1, ax1 = plt.subplots()
ax1.bar(["Logistic Regression"], [lr_acc])
ax1.set_ylabel("Accuracy Score")
ax1.set_title("Logistic Regression Performance")
ax1.set_ylim(0.5, 1.0)
st.pyplot(fig1)

# Decision Tree Chart
fig2, ax2 = plt.subplots()
ax2.bar(["Decision Tree"], [dt_acc])
ax2.set_ylabel("Accuracy Score")
ax2.set_title("Decision Tree Performance")
ax1.set_ylim(0.5, 1.0)
st.pyplot(fig2)

# Random Forest Chart
fig3, ax3 = plt.subplots()
ax3.bar(["Random Forest"], [rf_acc])
ax3.set_ylabel("Accuracy Score")
ax3.set_title("Random Forest Performance")
ax1.set_ylim(0.5, 1.0)
st.pyplot(fig3)


# Final Comparison Chart
st.subheader("📈 Model Comparison")

models = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [lr_acc, dt_acc, rf_acc]

fig4, ax4 = plt.subplots()
ax4.bar(models, accuracies)
ax4.set_ylabel("Accuracy Score")
ax4.set_title("Comparison of ML Models")

st.pyplot(fig4)