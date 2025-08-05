import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 data points
n_samples = 1000

# Generate features for class 0 (smaller values, more spread)
n_class_0 = 500
X_class_0 = np.random.normal(loc=1.0, scale=1.5, size=n_class_0)

# Generate features for class 1 (larger values, more spread)
n_class_1 = 500
X_class_1 = np.random.normal(loc=5.0, scale=1.2, size=n_class_1)

# Combine features and labels
X = np.concatenate([X_class_0, X_class_1])
y = np.concatenate([np.zeros(n_class_0), np.ones(n_class_1)])

# Shuffle the data
shuffle_idx = np.random.permutation(n_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]

# Create DataFrame
df = pd.DataFrame({'X': X, 'y': y})

# Save to CSV
df.to_csv('logistic_regression_data_1000.csv', index=False)

print(f"Generated {n_samples} data points!")
print(f"Class 0: {np.sum(y == 0)} samples")
print(f"Class 1: {np.sum(y == 1)} samples")
print(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))
print(f"\nDataset saved as 'logistic_regression_data_1000.csv'")
