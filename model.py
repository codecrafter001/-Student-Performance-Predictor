import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create sample dataset
np.random.seed(42)

data = {
    "study_hours": np.random.randint(1, 10, 200),
    "attendance": np.random.randint(60, 100, 200),
    "previous_score": np.random.randint(40, 90, 200),
    "final_score": np.random.randint(45, 100, 200)
}

df = pd.DataFrame(data)

X = df.drop("final_score", axis=1)
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("✅ Model trained & saved as model.pkl")
