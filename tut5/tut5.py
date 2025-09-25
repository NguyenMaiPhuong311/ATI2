import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
# Step 2. Identify inputs and outputs
df = pd.read_csv("ccpp.csv")

#  Rename columns to improve readability
df.rename(columns={'AT': 'Average Temperature', 
                   'V': 'Exhaust Vacuum',
                   'AP': 'Ambient Pressure',
                   'RH': 'Relative Humidity',
                   'PE': 'Net Hourly Electrical Energy Output'}, inplace=True)

# Inspect data for missing values and data shape
print("Number of rows and columns in the dataset: ", df.shape)
print("Missing values in each column: \n", df.isnull().sum())

# Descriptive statistics
print("\nDataset Info:")
print(df.info())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Correlation Analysis
print("\nCorrelation between features and target:")
correlation = df.corr()
print(correlation["Net Hourly Electrical Energy Output"].sort_values(ascending=False))

# Visualization: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Visualize relationships between features and target using pairplot
sns.pairplot(df,
             markers=".",
             kind='reg',
             diag_kind="auto",
             plot_kws={'line_kws':{'color':'#aec6cf'},
                       'scatter_kws': {'alpha': 0.5, 'color': '#82ad32'}},
             diag_kws={'color': '#82ad32'})
plt.show()

# Input and Output Selection
# From the heatmap and pairplot, we select the following features as inputs:
# - Average Temperature, Exhaust Vacuum, Ambient Pressure, Relative Humidity
# The target of the regression task is:
# - Net Hourly Electrical Energy Output (PE)

inputs = df[['Average Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']]
target = df['Net Hourly Electrical Energy Output']

# Step 3:
predictors = df.drop("Net Hourly Electrical Energy Output", axis=1).values
targets = df['Net Hourly Electrical Energy Output'].values

#  Split data into training and test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Check the shape of the training and test data
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Standardizing input data
scaler = StandardScaler()

# Fit and transform the scaler on training data, then transform the test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the normalized training and test data to inspect
print("Normalized training data:\n", X_train[:5])
print("Normalized test data:\n", X_test[:5])

# Step 4:
# Define the model
def get_new_model():
    # Initialize the model
    model = Sequential()
    
    # Add the first hidden layer with 6 neurons and ReLU activation function
    model.add(Dense(6, activation='relu'))
    
    # Add the second hidden layer with 6 neurons and ReLU activation function
    model.add(Dense(6, activation='relu'))
    
    # Output layer with one neuron (no activation function as this is a regression task)
    model.add(Dense(1))
    
    return model

# Step 5:
# Specify the model
model = get_new_model()

# Compile the model with the Adam optimizer and MSE (Mean Squared Error) loss
model.compile(optimizer='adam', loss="mean_squared_error")

# Define early stopping monitor to stop training if no improvement is seen in 4 epochs
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=4)

# Fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping_monitor])

# View training history
print(history.history)

# Step 6:
# Plot the loss over epochs to visualize the training process
plt.style.use("ggplot")
pd.DataFrame(history.history).plot(figsize=(12,10))

plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Step 7:
# Perform predictions using the trained model
predictions = model.predict(X_test)

# Plot the predicted values vs actual values
plt.figure(figsize=(12, 10))
sns.scatterplot(x=np.ravel(predictions), y=y_test)
plt.title("The Scatterplot of Relationship between Actual Values and Predictions")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.show()

# Step 8:


# Calculate MAE, MSE, and RMSE
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Evaluate explained variance score
print("Explained Variance Score:", metrics.explained_variance_score(y_test, predictions))
