## Overview
This project is on the implementation of Principal Component Analysis (PCA) and Logistic Regression using the Breast Cancer dataset. The dataset is sourced from the Scikit-learn library and contains various features related to breast cancer diagnosis.

## Requirements
Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

The Python script can be run on any Python IDE.

You can install these packages using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Dataset
The Breast Cancer Wisconsin (Diagnostic) dataset consists of:
- **569 instances**
- **30 numeric features** (e.g., mean radius, mean texture)
- **Target variable**: Class labels (malignant or benign)

### Loading the Dataset
You can load the dataset using the following code:
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

## Data Exploration
To understand the dataset, you can print the data description and summary statistics:
```python
print(data.DESCR)
```

## Data Preprocessing
### Scaling the Data
Since the features have different scales, it's essential to standardize them:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)
```

## Principal Component Analysis (PCA)
### Implementing PCA
1. **Without Dimensionality Reduction**:
   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=30)
   X_pca = pca.fit_transform(X_scaled)
   ```

2. **With Dimensionality Reduction to 2 Components**:
   ```python
   pca_2 = PCA(n_components=2)
   X1_pca = pca_2.fit_transform(X_scaled)
   ```

### Visualizing PCA Results
To visualize the PCA results:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X1_pca[:, 0], X1_pca[:, 1], c=data.target, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(label='Target')
plt.show()
```

## Logistic Regression
### Model Training
To train a Logistic Regression model:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1_pca, data.target, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```

### Making Predictions
To make predictions and evaluate the model:
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = log_reg.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
```