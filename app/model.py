from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model.fit(X, y)


def predict(value: float) -> float:
    prediction = model.predict([[value]])
    return float(prediction[0])
