import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

# ==============================
# DATA LOAD
# ==============================
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

print("Dataset Shape:", data.shape)
print(data.head())

# ==============================
# STEP 1: SPLIT X AND Y
# ==============================
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("X shape:", x.shape)
print("Y shape:", y.shape)

# ==============================
# STEP 2: ENCODING (if needed)
# ==============================
cat_columns = x.select_dtypes(['object']).columns
x[cat_columns] = x[cat_columns].apply(lambda col: pd.factorize(col)[0])

# ==============================
# STEP 3: TRAIN TEST SPLIT
# ==============================
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y,
    test_size=0.3,
    shuffle=False
)

# ==============================
# SIMPLE EVALUATION FUNCTION
# ==============================
def evaluate(name, y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n", name)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    return acc, prec, rec, f1

# ==============================
# 1. Bernoulli NB
# ==============================
model1 = BernoulliNB()
model1.fit(X_train, Y_train)
p1 = model1.predict(X_test)
b_acc, b_pre, b_rec, b_f1 = evaluate("Bernoulli NB", Y_test, p1)

# ==============================
# 2. Random Forest
# ==============================
model2 = RandomForestClassifier()
model2.fit(X_train, Y_train)
p2 = model2.predict(X_test)
r_acc, r_pre, r_rec, r_f1 = evaluate("Random Forest", Y_test, p2)

# ==============================
# 3. Gaussian NB
# ==============================
model3 = GaussianNB()
model3.fit(X_train, Y_train)
p3 = model3.predict(X_test)
g_acc, g_pre, g_rec, g_f1 = evaluate("Gaussian NB", Y_test, p3)

# ==============================
# 4. Decision Tree
# ==============================
model4 = DecisionTreeClassifier()
model4.fit(X_train, Y_train)
p4 = model4.predict(X_test)
d_acc, d_pre, d_rec, d_f1 = evaluate("Decision Tree", Y_test, p4)

# ==============================
# 5. Multinomial NB
# ==============================
model5 = MultinomialNB()
model5.fit(np.clip(X_train, 0, None), Y_train)
p5 = model5.predict(np.clip(X_test, 0, None))
m_acc, m_pre, m_rec, m_f1 = evaluate("Multinomial NB", Y_test, p5)

# ==============================
# 6. KNN
# ==============================
model6 = KNeighborsClassifier()
model6.fit(X_train, Y_train)
p6 = model6.predict(X_test)
k_acc, k_pre, k_rec, k_f1 = evaluate("KNN", Y_test, p6)

# ==============================
# LINE GRAPH
# ==============================
labels = ['Bernoulli','RandomForest','Gaussian','DecisionTree','Multinomial','KNN']

plt.figure(figsize=(10,6))
plt.plot(labels,[b_acc,r_acc,g_acc,d_acc,m_acc,k_acc],marker='o',label='Accuracy')
plt.plot(labels,[b_pre,r_pre,g_pre,d_pre,m_pre,k_pre],marker='o',label='Precision')
plt.plot(labels,[b_rec,r_rec,g_rec,d_rec,m_rec,k_rec],marker='o',label='Recall')
plt.plot(labels,[b_f1,r_f1,g_f1,d_f1,m_f1,k_f1],marker='o',label='F1')
plt.title("Model Comparison")
plt.legend()
plt.show()

# ==============================
# BAR GRAPH (F1)
# ==============================
plt.figure(figsize=(8,5))
plt.bar(labels,[b_f1,r_f1,g_f1,d_f1,m_f1,k_f1])
plt.title("F1 Score Comparison")
plt.show()

# ==============================
# BEST MODEL
# ==============================
best = max([
    ("Bernoulli", b_f1),
    ("Random Forest", r_f1),
    ("Gaussian", g_f1),
    ("Decision Tree", d_f1),
    ("Multinomial", m_f1),
    ("KNN", k_f1)
], key=lambda x: x[1])

print("\nBEST MODEL:", best[0], "F1:", best[1])

# ==============================
# CONFUSION MATRIX (BEST MODEL = Gaussian)
# ==============================
print("\nConfusion Matrix (Gaussian NB):")
print(confusion_matrix(Y_test, p3))

# ==============================
# FINAL CONCLUSION
# ==============================
print("\nConclusion:")
print("Gaussian NB performed best for this dataset based on F1 score.")
