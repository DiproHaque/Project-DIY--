# Define class weights: Give more importance to fraud (class 1)
class_weight = {0: 1., 1: 10.}  # 0 = non-fraud, 1 = fraud

from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X = dsf.drop('is_weekend', axis=1)  # Drop the target column for features
y = dsf['is_weekend']  # Target column

# Split into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the split sizes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# Make copies to avoid modifying the original X_train and X_test
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Ensure all categorical columns, including 'day_name', are encoded
# Create a temporary list of categorical columns for encoding in this cell
cols_to_encode = list(cat_cols) # Start with the existing cat_cols
if 'day_name' in X_train_encoded.columns and 'day_name' not in cols_to_encode:
    cols_to_encode.append('day_name')

# Apply Label Encoding to categorical columns
for col in cols_to_encode:
    if col in X_train_encoded.columns:
        le = LabelEncoder()
        # Handle potential NaNs before encoding if necessary, or fill them
        X_train_encoded[col] = X_train_encoded[col].astype(str).fillna('NaN_val') # Convert to string to handle mixed types gracefully
        X_test_encoded[col] = X_test_encoded[col].astype(str).fillna('NaN_val')

        # Fit on combined data to ensure consistency across train and test
        le.fit(pd.concat([X_train_encoded[col], X_test_encoded[col]], axis=0).astype(str)) # Fit on combined data after converting to string
        X_train_encoded[col] = le.transform(X_train_encoded[col])
        X_test_encoded[col] = le.transform(X_test_encoded[col])

# Define the Autoencoder architecture
input_dim = X_train_encoded.shape[1]  # Number of features in the data after encoding

# Define the encoding layer
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)

# Define the decoding layer
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Create the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the model
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the Autoencoder using the encoded data
autoencoder.fit(X_train_encoded, X_train_encoded, epochs=50, batch_size=256, validation_data=(X_test_encoded, X_test_encoded), verbose=1)

import numpy as np

# Predict the reconstructed values (outputs of the Autoencoder)
X_train_pred = autoencoder.predict(X_train_encoded)
X_test_pred = autoencoder.predict(X_test_encoded)

# Calculate the Mean Squared Error (MSE) for each transaction
train_mse = np.mean(np.power(X_train_encoded - X_train_pred, 2), axis=1)
test_mse = np.mean(np.power(X_test_encoded - X_test_pred, 2), axis=1)

# Set a threshold for anomaly detection (e.g., 95th percentile of MSE)
threshold = np.percentile(train_mse, 95)

# Flag anomalies (transactions with MSE greater than the threshold)
y_train_pred = train_mse > threshold
y_test_pred = test_mse > threshold

# Evaluate the model
from sklearn.metrics import classification_report

print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# Displaying the details of the fraud cases in the training and test sets
fraud_train_cases_details = X_train[y_train_pred == 1][['USER_ID', 'TXN_AMOUNT']]
fraud_test_cases_details = X_test[y_test_pred == 1][['USER_ID', 'TXN_AMOUNT']]

# Print first few rows of fraud cases details from training set
print("Fraud cases in training data:")
print(fraud_train_cases_details)

# Print first few rows of fraud cases details from test set
print("Fraud cases in test data:")
print(fraud_test_cases_details)

# Save fraud cases from training data to a CSV file
fraud_train_cases_details.to_csv('fraud_train_cases.csv', index=False)

# Save fraud cases from test data to a CSV file
fraud_test_cases_details.to_csv('fraud_test_cases.csv', index=False)

print("Fraud cases from training and test data have been saved as CSV files.")
