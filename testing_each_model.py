import joblib

# Load the logistic regression model from the file
logistic_regression_model = joblib.load('models/logistic_regression_model.pkl')

# Load the decision tree model from the file
decision_tree_model = joblib.load('models/decision_tree_model.pkl')

# Load the random forest model from the file
random_forest_model = joblib.load('models/random_forest_model.pkl')

# Load the neural network model from the file
neural_network_model = joblib.load('models/neural_network_model.pkl')



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Load the data into a Pandas DataFrame
data=pd.read_csv("creditcard.csv")
data.head()
# Separate features and target variable
X=data.drop(['Class'],axis=1).values
Y=data['Class'].values
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(X,Y,test_size=0.2,random_state=1000)

logistic_regression_predictions = logistic_regression_model.predict(xTest)
# Evaluate the predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(yTest, logistic_regression_predictions)
precision = precision_score(yTest, logistic_regression_predictions)
recall = recall_score(yTest, logistic_regression_predictions)
f1 = f1_score(yTest, logistic_regression_predictions)
roc_auc = roc_auc_score(yTest, logistic_regression_model.predict_proba(xTest)[:, 1])
print("Logistic Regression Model Metrics:")
print("Accuracy (in %):", accuracy*100)
print("Precision (in %):", precision*100)
print("Recall (in %):", recall*100)
print("F1 Score (in %):", f1*100)
print("ROC AUC Score (in %):", roc_auc*100)

decision_tree_predictions = decision_tree_model.predict(xTest)
# Evaluate the predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(yTest, decision_tree_predictions)
precision = precision_score(yTest, decision_tree_predictions)
recall = recall_score(yTest, decision_tree_predictions)
f1 = f1_score(yTest, decision_tree_predictions)
roc_auc = roc_auc_score(yTest, decision_tree_model.predict_proba(xTest)[:, 1])
print("Decision Tree Model Metrics:")
print("Accuracy (in %):", accuracy*100)
print("Precision (in %):", precision*100)
print("Recall (in %):", recall*100)
print("F1 Score (in %):", f1*100)
print("ROC AUC Score (in %):", roc_auc*100)

random_forest_predictions = random_forest_model.predict(xTest)
# Evaluate the predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(yTest, random_forest_predictions)
precision = precision_score(yTest, random_forest_predictions)
recall = recall_score(yTest, random_forest_predictions)
f1 = f1_score(yTest, random_forest_predictions)
roc_auc = roc_auc_score(yTest, random_forest_model.predict_proba(xTest)[:, 1])
print("Random Forest Model Metrics:")
print("Accuracy (in %):", accuracy*100)
print("Precision (in %):", precision*100)
print("Recall (in %):", recall*100)
print("F1 Score (in %):", f1*100)
print("ROC AUC Score (in %):", roc_auc*100)

neural_network_predictions = neural_network_model.predict(xTest)
neural_network_predictions_bool_as_int = (neural_network_predictions > 0.5).astype(int)
# Evaluate the predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(yTest, neural_network_predictions_bool_as_int)
if any(neural_network_predictions_bool_as_int == 1):
    precision = precision_score(yTest, neural_network_predictions_bool_as_int)
else:
    precision = 0.0
precision = precision_score(yTest, neural_network_predictions_bool_as_int)
recall = recall_score(yTest, neural_network_predictions_bool_as_int)
f1 = f1_score(yTest, neural_network_predictions_bool_as_int)
roc_auc = roc_auc_score(yTest, neural_network_predictions)
print("Neural Network Model Metrics:")
print("Accuracy (in %):", accuracy*100)
print("Precision (in %):", precision*100)
print("Recall (in %):", recall*100)
print("F1 Score (in %):", f1*100)
print("ROC AUC Score (in %):", roc_auc*100)