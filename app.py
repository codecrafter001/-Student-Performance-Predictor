from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        reading = int(request.form["reading"])
        writing = int(request.form["writing"])
        test_prep = int(request.form["test_prep"])
        parent_edu = int(request.form["parent_edu"])

        features = np.array([[reading, writing, test_prep, parent_edu]])
        prediction = round(model.predict(features)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
