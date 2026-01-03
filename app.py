from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ------------------------------------
# Train model directly (NO model.pkl)
# ------------------------------------
def train_model():
    df = pd.read_csv("student_performance_data.csv")

    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["final_score"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Train once when app starts
model = train_model()

# ------------------------------------
# Routes
# ------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])

        features = np.array([[study_hours, attendance, previous_score]])
        prediction = round(model.predict(features)[0], 2)

    return render_template("index.html", prediction=prediction)

# ------------------------------------
# Run
# ------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
