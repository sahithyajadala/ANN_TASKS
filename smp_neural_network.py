import numpy as np
import os
import cv2

# Load training data
training_path = "/home/sahitya-jadala/Downloads/1st_week_project/train/"
images = []
labels = []
for root, dirs, files in os.walk(training_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        label = os.path.basename(root)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (28, 28))
            images.append(image.flatten())
            labels.append(label)
X_train = np.array(images)
y_train = np.array(labels)

# Map labels to integers
label_to_int = {label: idx for idx, label in enumerate(np.unique(y_train))}
y_train = np.array([label_to_int[label] for label in y_train])

# Initialize perceptron
input_size = X_train.shape[1]
weights = np.zeros(input_size)
bias = 0
learning_rate = 0.1
epochs = 1

# Train perceptron
for epoch in range(epochs):
    for i in range(len(X_train)):
        prediction = np.dot(X_train[i], weights) + bias
        prediction = 1 if prediction >= 0 else 0
        if prediction != y_train[i]:
            update = learning_rate * (y_train[i] - prediction)
            weights += update * X_train[i]
            bias += update

# Evaluate training accuracy
correct = 0
for i in range(len(X_train)):
    prediction = np.dot(X_train[i], weights) + bias
    prediction = 1 if prediction >= 0 else 0
    if prediction == y_train[i]:
        correct += 1
train_acc = correct / len(X_train)
print(f"Training Accuracy: {train_acc:.2f}")

# Optionally, load validation data and evaluate accuracy
validation_path = "/home/sahitya-jadala/Downloads/1st_week_project/valid/"
images = []
labels = []
for root, dirs, files in os.walk(validation_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        label = os.path.basename(root)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (28, 28))
            images.append(image.flatten())
            labels.append(label)
X_val = np.array(images)
y_val = np.array(labels)

label_to_int = {label: idx for idx, label in enumerate(np.unique(y_val))}
y_val = np.array([label_to_int[label] for label in y_val])

correct = 0
for i in range(len(X_val)):
    prediction = np.dot(X_val[i], weights) + bias
    prediction = 1 if prediction >= 0 else 0
    if prediction == y_val[i]:
        correct += 1
val_acc = correct / len(X_val)
print(f"Validation Accuracy: {val_acc:.2f}")

# Optionally, load testing data and evaluate accuracy
testing_path = "/home/sahitya-jadala/Downloads/1st_week_project/test/"
images = []
labels = []
for root, dirs, files in os.walk(testing_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        label = os.path.basename(root)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (28, 28))
            images.append(image.flatten())
            labels.append(label)
X_test = np.array(images)
y_test = np.array(labels)

label_to_int = {label: idx for idx, label in enumerate(np.unique(y_test))}
y_test = np.array([label_to_int[label] for label in y_test])

correct = 0
for i in range(len(X_test)):
    prediction = np.dot(X_test[i], weights) + bias
    prediction = 1 if prediction >= 0 else 0
    if prediction == y_test[i]:
        correct += 1
test_acc = correct / len(X_test)
print(f"Testing Accuracy: {test_acc:.2f}")