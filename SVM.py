import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("C:\\Users\\harsh\\Downloads\\updated_nipah_virus_symptoms.csv")
df.head()
print(df.isnull().sum())

df['Fever'].fillna(df['Fever'].median(), inplace=True)
df['Headache'].fillna(df['Headache'].median(), inplace=True)
df['Dizziness'].fillna(df['Dizziness'].median(), inplace=True)
df['Respiratory Symptoms'].fillna(df['Respiratory Symptoms'].median(), inplace=True)
df['Altered Consciousness'].fillna(df['Altered Consciousness'].median(), inplace=True)
print(df.isnull().sum())

#Separate features (X) and target variable (y)
X = df[['Fever', 'Headache', 'Dizziness', 'Respiratory Symptoms', 'Seizures', 'Altered Consciousness', 'Nausea/Vomiting']]
y = df['NIV infected']

 #Scale the features to a range (0, 1) for better SVM performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the SVM model
# 'linear' kernel for linear SVM, but 'rbf' (Radial Basis Function) is more common for non-linear classification
model = SVC(kernel='linear', random_state=42)  # You can try 'rbf' kernel as well
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model's accuracy confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Print first 5 actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# Step 8: Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()