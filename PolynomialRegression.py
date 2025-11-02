# TipsPredict/PolynomialRegression.py
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "tip.csv")
print("Loading CSV from:", csv_path)

# Load the dataset using pandas
data_full = pd.read_csv(csv_path)
data = data_full.iloc[:200].copy() # Use only the first 200 rows for simplicity

# Extract features and target variable
X = data[['total_bill','size']].values # Numeric data for features
y = data['tip'].values.reshape(-1, 1) 
m = len(y) 

# Create polynomial features (squared terms)
total_bill_sq = (data['total_bill'] ** 2).values.reshape(-1, 1)
size_sq = (data['size'] ** 2).values.reshape(-1, 1)
interaction = (data['total_bill'] * data['size']).values.reshape(-1, 1) # Feature Engineering: interaction term

# Combine original and polynomial features
X_poly_A = np.hstack((X, total_bill_sq, size_sq, interaction)) 
X_poly_B = np.hstack((X, total_bill_sq)) # Simpler polynomial features

# Scale features
def feature_scaling(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    return X_scaled, X_mean, X_std
X_poly_A_scaled, X_mean_A, X_std_A = feature_scaling(X_poly_A)
X_poly_B_scaled, X_mean_B, X_std_B = feature_scaling(X_poly_B)

# Add intercept term (only after scaling)
X_poly_A_scaled = np.hstack((np.ones((m, 1)), X_poly_A_scaled))
X_poly_B_scaled = np.hstack((np.ones((m, 1)), X_poly_B_scaled))

# Gradient Descent Implementation
def gradient_descent(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape # m = number of examples, n = number of features
    w = np.zeros((n, 1)) # Weight Vector
    cost_history = [] 
    for epoch in range(epochs):
        y_pred = X.dot(w)
        error = y_pred - y

        gradient = (1/m) * X.T.dot(error)
        w -= alpha * gradient

        cost = (1/(2*m)) * np.sum(error ** 2)
        cost_history.append(cost)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Cost: {cost}')
    return w, cost_history

# Train models
w_A, cost_history_A = gradient_descent(X_poly_A_scaled, y, alpha=0.01, epochs=1000)
w_B, cost_history_B = gradient_descent(X_poly_B_scaled, y, alpha=0.01, epochs=1000)

# Plot cost history
plt.plot(cost_history_A, label='Model A (More Features)', color='red', linestyle='-', linewidth=2)
plt.plot(cost_history_B, label='Model B (Fewer Features)', color='green', linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Reduction over Time')
plt.legend()
plt.show()

# Predict function
def predict(X, w, X_mean, X_std):
    X_scaled = (X - X_mean) / X_std
    X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
    return X_scaled.dot(w)

# Create a 3D plot to visualize Model A predictions
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(data['total_bill'], data['size'], data['tip'], color='blue', label='Actual Tips', alpha=0.6)

# Generate a grid of values for visualization
bill_range = np.linspace(data['total_bill'].min(), data['total_bill'].max(), 30)
size_range = np.linspace(data['size'].min(), data['size'].max(), 30)
bill_grid, size_grid = np.meshgrid(bill_range, size_range)

# Create polynomial features for the grid (same as model A)
bill_sq = bill_grid ** 2
size_sq = size_grid ** 2
interaction_grid = bill_grid * size_grid

X_grid = np.column_stack((
    bill_grid.ravel(),
    size_grid.ravel(),
    bill_sq.ravel(),
    size_sq.ravel(),
    interaction_grid.ravel()
))

# Predict tips for the grid
pred_grid = predict(X_grid, w_A, X_mean_A, X_std_A).reshape(bill_grid.shape)

# Plot surface
ax.plot_surface(bill_grid, size_grid, pred_grid, color='orange', alpha=0.5, label='Model A Surface')

ax.set_xlabel('Total Bill')
ax.set_ylabel('Size')
ax.set_zlabel('Tip')
ax.set_title('Polynomial Regression Surface (Model A)')
plt.legend()
plt.show()

# Example prediction
new_data_df = data_full.iloc[200:205][['total_bill','size']] # New data for prediction
new_data = new_data_df.values

new_total_bill_sq = (new_data_df['total_bill'] ** 2).values.reshape(-1, 1)
new_size_sq = (new_data_df['size'] ** 2).values.reshape(-1, 1)
new_interaction = (new_data_df['total_bill'] * new_data_df['size']).values.reshape(-1, 1)


new_X_poly_A = np.hstack((new_data, new_total_bill_sq, new_size_sq, new_interaction))
new_X_poly_B = np.hstack((new_data, new_total_bill_sq))

preds_A = predict(new_X_poly_A, w_A, X_mean_A, X_std_A)
preds_B = predict(new_X_poly_B, w_B, X_mean_B, X_std_B)

print("Predictions from Model A (More Features):", preds_A)
print("Predictions from Model B (Fewer Features):", preds_B)



# Conclusion:
# Tips increase roughly linearly with the total bill, with some influence from the group size
# The relationship between the features and the target is mostly linear.
# Polynomial terms (total_bill², size², interaction) only slightly improved the model’s fit.
# Thus, the model can’t capture complex, real-world variations (like smokers tipping differently at dinner)


# Tipping is noisy and categorical factors matter more