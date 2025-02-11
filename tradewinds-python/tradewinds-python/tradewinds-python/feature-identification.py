import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SVMSMOTE, SMOTE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets from CSV files
X = pd.read_csv('/Users/brianmewhinney/dev/tradewinds-python/data/8/x.csv').values
y = pd.read_csv('/Users/brianmewhinney/dev/tradewinds-python/data/8/y.csv').values

# One-hot encode the y labels
y = np.argmax(to_categorical(y, num_classes=3), axis=1)  # Convert for SMOTE

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance with SVM-SMOTE
svm_smote = SVMSMOTE(random_state=42)
X_resampled_svm, y_resampled_svm = svm_smote.fit_resample(X_train, y_train)

# Address class imbalance with SMOTE-NN
smote_nn = SMOTE(random_state=42, k_neighbors=3)
X_resampled_smote, y_resampled_smote = smote_nn.fit_resample(X_train, y_train)

# Standardize the features
#scaler = StandardScaler()
scaler = MinMaxScaler()
X_resampled_svm = scaler.fit_transform(X_resampled_svm)
X_resampled_smote = scaler.transform(X_resampled_smote)
X_test = scaler.transform(X_test)

# One-hot encode the resampled labels for neural network
y_resampled_svm = to_categorical(y_resampled_svm, num_classes=3)
y_resampled_smote = to_categorical(y_resampled_smote, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Function to create and train model
def create_and_train_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')
    
    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()


# Train and evaluate the model with SVM-SMOTE resampled data
print("Training with SVM-SMOTE resampled data:")
create_and_train_model(X_resampled_svm, y_resampled_svm, X_test, y_test)

# Train and evaluate the model with SMOTE-NN resampled data
print("Training with SMOTE-NN resampled data:")
create_and_train_model(X_resampled_smote, y_resampled_smote, X_test, y_test)