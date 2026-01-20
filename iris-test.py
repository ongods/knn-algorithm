from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)

# Print
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Classifier on Iris Dataset Confusion Matrix")
plt.show()