import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Split X and Y
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# Train Gaussian NB (best model from Lab 11)
model = GaussianNB()
model.fit(X_train, Y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl saved successfully!")
print("Test Accuracy:", round(model.score(X_test, Y_test) * 100, 2), "%")
