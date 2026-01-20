from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("Dataset Information")
print(f"Total Samples {len(X)}")
print(f"Number of Features: {X.shape[1]}")
print(f"Malignant Cases: {sum(y==0)}")
print(f"Benign Cases: {sum(y==1)}")
print("\n" + "="*50 + "\n")

# Split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Make predictions 
predictions = knn.predict(X_test_scaled)
accuracy = knn.score(X_test_scaled, y_test)

print(f"KNN Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n" + "="*50 + "\n")

print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Malignant', 'Benign']))

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('KNN Confusion Matrix for Breast Cancer Detection')
plt.tight_layout()
plt.show()

# Accuracy vs K-value
print("\n Testing different K-values for KNN Classifier")
accuracies = []
k_values = range(1, 31)

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_temp.fit(X_train_scaled, y_train)
    accuracies.append(knn_temp.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.title('KNN Performance: Accuracy vs K value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=max(accuracies), color='r', linestyle='--', alpha=0.5, label=f'Best: K={k_values[accuracies.index(max(accuracies))]} ({max(accuracies):.4f})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nBest K value: {k_values[accuracies.index(max(accuracies))]} with accuracy: {max(accuracies):.4f}")