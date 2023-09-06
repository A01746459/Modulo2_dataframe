import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
dataset = pd.read_csv("KNNAlgorithmDataset.csv", delimiter=",")
dataset = dataset.drop(columns=["id", "Unnamed: 32"], axis=1)
dataset = dataset.dropna()

dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
X = dataset.drop('diagnosis', axis=1)
y = dataset['diagnosis']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Step 6: Standardize the features (optional but recommended for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Create a KNN classifier and train it
knn = KNeighborsClassifier(n_neighbors=7)  # You can adjust the number of neighbors (k) as needed
knn.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = knn.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", classification_report_str)