# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Read the csv file
main_df = pd.read_csv('/alzheimers_prediction_dataset.csv') #insert the path to the csv
# main_df.info()
# main_df.head()
# main_df.columns
df = main_df

#Encode object datatypes to int datatypes to fit into the model
le = LabelEncoder()
for columns in df.columns:
  if df[columns].dtypes == "object":
    df[columns] = le.fit_transform(df[columns])

# df.head()
#Check for missing data
df.isnull().sum()

# Drop unnecessary columns
df.drop(["Country","Education Level","Employment Status", "Marital Status","Social Engagement Level","Income Level","Urban vs Rural Living"], axis="columns", inplace = True)
# df.head()

# Split the dataset to train and test the model
X_train, X_test, y_train, y_test = train_test_split(df.drop("Alzheimer’s Diagnosis", axis="columns"), df["Alzheimer’s Diagnosis"], test_size=0.2, random_state=30)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Assign model and fit train data
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true valuse')
plt.ylabel('predicted values');


# Assign another model
model2 = LogisticRegression()
model2.fit(X_train, y_train)
y_pred1 = model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))


# Plot the confusion matrix
mat = confusion_matrix(y_test, y_pred1)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true valuse')
plt.ylabel('predicted values');
