import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Loading the required dataset
df = pd.read_csv('/content/nipah_virus_symptoms.csv')
df.head()
# to get the total sum of missing values in each column in our dataset
print(df.isnull().sum())
# Avoiding chained assignment by updating columns directly
df['Fever'] = df['Fever'].fillna(df['Fever'].median())
df['Headache'] = df['Headache'].fillna(df['Headache'].median())
df['Dizziness'] = df['Dizziness'].fillna(df['Dizziness'].median())
df['Respiratory Symptoms'] = df['Respiratory Symptoms'].fillna(df['Respiratory Symptoms'].median())
df['Altered Consciousness'] = df['Altered Consciousness'].fillna(df['Altered Consciousness'].median())

# Checking for any remaining missing values
print(df.isnull().sum())
# Splitting data into training and testing sets
X = df[['Fever', 'Headache', 'Dizziness', 'Respiratory Symptoms', 'Seizures', 'Altered Consciousness', 'Nausea/Vomiting']]
y = df['NIV infected']
# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the ANN
model = Sequential()
# Input layer and first hidden layer (example: 10 neurons)
model.add(Dense(units=10, activation='relu', input_shape=(X_train.shape[1],)))
# Second hidden layer (optional: can add more layers if needed)
model.add(Dense(units=8, activation='relu'))
# Output layer (binary classification)
model.add(Dense(units=1, activation='sigmoid'))
# Compile the model using binary cross-entropy loss since this is a binary classification problem and by also using Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the ANN
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary (0 or 1)
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy}")
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()