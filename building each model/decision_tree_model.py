import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import gridspec

# Load the data into a Pandas DataFrame
data=pd.read_csv("creditcard.csv")

data.head()

fraud=data[data['Class']==1]
valid=data[data['Class']==0]
outlierFraction=len(fraud)/float(len(valid))

fraud.Amount.describe()

valid.Amount.describe()

# Separate features and target variable
X=data.drop(['Class'],axis=1).values
Y=data['Class'].values

# Create a StandardScaler instance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(X,Y,test_size=0.2,random_state=1000)

# Scale the training data
X_train_scaled = scaler.fit_transform(xTrain)

# Create an instance of SMOTEENN (SMOTE + ENN)
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=42)

# Apply SMOTEENN to the training data
X_train_resampled, y_train_resampled = smoteenn.fit_resample(xTrain, yTrain)

# Apply SMOTEENN to the scaled training data
X_train_resampled_scaled, y_train_resampled_scaled = smoteenn.fit_resample(X_train_scaled, yTrain)

#Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier
# Create a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model using the training data
dt_model.fit(xTrain,yTrain) # Using the unprocessed data resulted in better values

# Make predictions on the test data
yPred_dt = dt_model.predict(xTest)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(yTest, yPred_dt)
precision = precision_score(yTest, yPred_dt)
recall = recall_score(yTest, yPred_dt)
f1 = f1_score(yTest, yPred_dt)
roc_auc = roc_auc_score(yTest, dt_model.predict_proba(xTest)[:, 1])

print("Decision Tree Model Metrics:")
print("Accuracy (in %):", accuracy*100)
print("Precision (in %):", precision*100)
print("Recall (in %):", recall*100)
print("F1 Score (in %):", f1*100)
print("ROC AUC Score (in %):", roc_auc*100)

import joblib

# Assuming you have trained a model named 'trained_model'
# Save the model to a file named 'model.pkl'
joblib.dump(dt_model, 'decision_tree_model.pkl')