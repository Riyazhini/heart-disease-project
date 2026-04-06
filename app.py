import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Heart Disease App ❤️", layout="wide")

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv("dataset/heart.csv")

X = data[["age", "trestbps", "chol", "thalach"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ---------------- #
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---------------- PATIENT STORAGE ---------------- #
FILE = "patients.csv"

if os.path.exists(FILE):
    patients = pd.read_csv(FILE)
else:
    patients = pd.DataFrame([
        {"Name": "Arun", "Age": 45, "BP": 130, "Chol": 250, "HR": 150},
        {"Name": "Bala", "Age": 50, "BP": 140, "Chol": 260, "HR": 140},
        {"Name": "Cathy", "Age": 38, "BP": 120, "Chol": 210, "HR": 160},
        {"Name": "David", "Age": 60, "BP": 150, "Chol": 300, "HR": 130},
        {"Name": "Elan", "Age": 42, "BP": 135, "Chol": 240, "HR": 155}
    ])
    patients.to_csv(FILE, index=False)

# ---------------- NAVIGATION ---------------- #
col1, col2, col3, col4, col5 = st.columns([5,1,1,1,1])

with col2:
    if st.button("🏠"):
        st.session_state.page = "Home"
with col3:
    if st.button("👥"):
        st.session_state.page = "Patients"
with col4:
    if st.button("🔍"):
        st.session_state.page = "Prediction"
with col5:
    if st.button("📈"):
        st.session_state.page = "Evaluation"

if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

# ---------------- HOME ---------------- #
if page == "Home":

    total = len(data)
    disease = len(data[data["target"] == 1])
    normal = len(data[data["target"] == 0])
    monitoring = len(data[(data["chol"] > 240) | (data["trestbps"] > 140)])

    st.title("❤️ Heart Disease Dashboard")

    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;'>
        <div style='background:#1f4037;color:white;padding:20px;border-radius:15px;width:23%;text-align:center;'>
            <h4>Total Patients</h4><h2>{total}</h2>
        </div>
        <div style='background:#ff4b4b;color:white;padding:20px;border-radius:15px;width:23%;text-align:center;'>
            <h4>Heart Disease</h4><h2>{disease}</h2>
        </div>
        <div style='background:#4CAF50;color:white;padding:20px;border-radius:15px;width:23%;text-align:center;'>
            <h4>Normal</h4><h2>{normal}</h2>
        </div>
        <div style='background:#ff9800;color:white;padding:20px;border-radius:15px;width:23%;text-align:center;'>
            <h4>Monitoring</h4><h2>{monitoring}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.container():
        st.subheader("📊 Dataset Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.hist(data["age"], bins=15)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

        with col2:
            counts = data["target"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.bar(["Normal", "Disease"], counts)
            ax2.set_title("Disease Distribution")
            st.pyplot(fig2)

# ---------------- PATIENTS ---------------- #
elif page == "Patients":

    st.title("👥 Patient Management")

    df = pd.read_csv(FILE)

    search = st.text_input("Search")
    if search:
        df = df[df["Name"].str.contains(search, case=False)]

    sort = st.selectbox("Sort", ["Ascending", "Descending"])
    df = df.sort_values("Name", ascending=(sort == "Ascending"))

    st.dataframe(df)

    st.subheader("➕ Add Patient")
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 100)
    bp = st.number_input("BP", 80, 200)
    chol = st.number_input("Chol", 100, 400)
    hr = st.number_input("HR", 60, 200)

    if st.button("Add"):
        new = pd.DataFrame([[name, age, bp, chol, hr]],
                           columns=df.columns)
        df = pd.concat([df, new], ignore_index=True)
        df.to_csv(FILE, index=False)
        st.success("Added!")

    st.subheader("✏️ Edit Patient")
    edit_name = st.selectbox("Select Patient", df["Name"])

    if edit_name:
        row = df[df["Name"] == edit_name].index[0]

        age = st.number_input("Edit Age", 1, 100, int(df.loc[row, "Age"]))
        bp = st.number_input("Edit BP", 80, 200, int(df.loc[row, "BP"]))
        chol = st.number_input("Edit Chol", 100, 400, int(df.loc[row, "Chol"]))
        hr = st.number_input("Edit HR", 60, 200, int(df.loc[row, "HR"]))

        if st.button("Update"):
            df.loc[row] = [edit_name, age, bp, chol, hr]
            df.to_csv(FILE, index=False)
            st.success("Updated!")

    st.subheader("🗑 Delete Patient")
    del_name = st.selectbox("Delete Patient", df["Name"])

    if st.button("Delete"):
        df = df[df["Name"] != del_name]
        df.to_csv(FILE, index=False)
        st.warning("Deleted!")

# ---------------- PREDICTION ---------------- #
elif page == "Prediction":

    st.title("🔍 Prediction")

    df = pd.read_csv(FILE)
    names = ["Manual"] + list(df["Name"])

    selected = st.selectbox("Select Patient", names)

    if selected == "Manual":
        st.subheader("✍️ Enter Patient Details")
        pname = st.text_input("Patient Name")
        age = st.slider("Age", 1, 100, 25)
        bp = st.slider("BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)
        hr = st.slider("Heart Rate", 60, 200, 100)
    else:
        p = df[df["Name"] == selected].iloc[0]
        pname = selected
        age, bp, chol, hr = p["Age"], p["BP"], p["Chol"], p["HR"]
        st.info(f"Selected Patient: {pname}")

    if st.button("Predict"):
        pred = rf.predict([[age, bp, chol, hr]])[0]

        st.subheader(f"Result for {pname}")

        if pred == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ Normal Condition")

# ---------------- EVALUATION ---------------- #
elif page == "Evaluation":

    st.title("📈 Model Evaluation")

    # Accuracy
    lr_pred = lr.predict(X_test)
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_pred)
    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    models = {
        "Logistic Regression": lr_acc,
        "Decision Tree": dt_acc,
        "Random Forest": rf_acc
    }

    best_model = max(models, key=models.get)
    best_score = models[best_model]

    st.markdown(f"""
    <div style='padding:15px;border-radius:10px;background:#e8f5e9;'>
    <h3>🏆 Best Model: {best_model}</h3>
    <h4>Accuracy: {round(best_score, 2)}</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, rf_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("### ROC Curve")
            y_prob = rf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr)
            ax2.plot([0,1],[0,1],'--')
            st.pyplot(fig2)

    st.markdown("---")

with st.container():
    st.subheader("📊 Model Comparison")

    # Use columns to control width
    col1, col2, col3 = st.columns([1, 2, 1])  # center the chart

    with col2:
        # Create the figure with controlled size
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.bar(list(models.keys()), list(models.values()), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0,1)  # optional: fix y-axis from 0 to 1
        ax3.set_title("Model Accuracy Comparison")

        st.pyplot(fig3)