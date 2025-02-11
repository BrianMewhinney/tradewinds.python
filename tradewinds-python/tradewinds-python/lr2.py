import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SVMSMOTE, SMOTE
from scikeras.wrappers import KerasClassifier  # Correct import



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
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_train)
X_resampled_svm = scaler.fit_transform(X_resampled_svm)
X_resampled_smote = scaler.transform(X_resampled_smote)
X_test = scaler.transform(X_test)

# One-hot encode the resampled labels for neural network
y_resampled = to_categorical(y_train, num_classes=3)
y_resampled_svm = to_categorical(y_resampled_svm, num_classes=3)
y_resampled_smote = to_categorical(y_resampled_smote, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Function to create the model
def create_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Input layer
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),  # First hidden layer
        BatchNormalization(),
        Dropout(0.5),  # Dropout for regularization
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),  # Second hidden layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),  # Third hidden layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),  # Fourth hidden layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Output layer
    ])
    
    optimizer = Adam(learning_rate=0.002)  # Set learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

# Train and evaluate the model with SVM-SMOTE resampled data
print("Training with SVM-SMOTE resampled data:")
create_and_train_model(X_resampled, y_resampled, X_test, y_test)

# Train and evaluate the model with SMOTE-NN resampled data
#print("Training with SMOTE-NN resampled data:")
#create_and_train_model(X_resampled_smote, y_resampled_smote, X_test, y_test)
