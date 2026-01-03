import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample dataset
np.random.seed(42)

data = {
    "reading_score": np.random.randint(40, 100, 300),
    "writing_score": np.random.randint(40, 100, 300),
    "test_prep": np.random.randint(0, 2, 300),   # 0 = No, 1 = Yes
    "parent_edu": np.random.randint(0, 3, 300),  # Encoded
    "math_score": np.random.randint(40, 100, 300)
}

df = pd.DataFrame(data)

X = df.drop("math_score", axis=1)
y = df["math_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("✅ Model trained & saved as model.pkl")
