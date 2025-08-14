# train_classifier.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

# Step 1: Load the saved embeddings and labels
embeddings = np.load("../data/embeddings.npy")
labels = np.load("../data/labels.npy")

# Step 2: Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Step 3: Train the classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Precision:", precision_score(y_test, y_pred))
print("✅ Recall:", recall_score(y_test, y_pred))

# Step 5: Save the trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(clf, "../models/classifier_model.pkl")

print("✅ Trained model saved to models/classifier_model.pkl")
