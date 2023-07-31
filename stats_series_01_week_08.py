# Snippet 1: Introduction to Logistic Regression
from sklearn.linear_model import LogisticRegression

X, y = [[1], [2], [3], [4]], [0, 0, 1, 1] # Simple example
logistic_model = LogisticRegression().fit(X, y)

# Snippet 2: Building and Interpreting a Logistic Regression Model
import numpy as np

probs = logistic_model.predict_proba(X)
odds_ratios = np.exp(logistic_model.coef_) # Calculating Odds Ratios

# Snippet 3: Evaluation Metrics for Classification
from sklearn.metrics import accuracy_score, confusion_matrix

predictions = logistic_model.predict(X)
accuracy = accuracy_score(y, predictions)
conf_matrix = confusion_matrix(y, predictions)

# Snippet 4: Applications and Real-world Examples (Using previous model)
# Note: This code builds on previous Snippets; apply the model to relevant real-world data
