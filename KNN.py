import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\harsh\\Downloads\\updated_nipah_virus_symptoms.csv")
df.head()
df['Fever'].fillna(df['Fever'].median(), inplace=True)
df['Headache'].fillna(df['Headache'].median(), inplace=True)
df['Dizziness'].fillna(df['Dizziness'].median(), inplace=True)
df['Respiratory Symptoms'].fillna(df['Respiratory Symptoms'].median(), inplace=True)
df['Altered Consciousness'].fillna(df['Altered Consciousness'].median(), inplace=True)
print(df.isnull().sum())
#Separate features (X) and target variable (y)
X = df[['Fever', 'Headache', 'Dizziness', 'Respiratory Symptoms', 'Seizures', 'Altered Consciousness', 'Nausea/Vomiting']]
y = df['NIV infected']
#Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the KNN model
# Initialize KNeighborsClassifier, k=5 (you can adjust k)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# Output results
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