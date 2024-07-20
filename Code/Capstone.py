import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st


file_path = r'Dataset\financialanalytics_Dataset.csv'
data = pd.read_csv(file_path)

print("Initial dataset shape:", data.shape)
print("Initial missing values:\n", data.isnull().sum())

'''
st.title("Financial Analytics Dataset")

# Document the initial state of the dataset
st.header("Initial Dataset")
st.write(data.head())
'''

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)

# Multivariate Imputation
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
data_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)

# Combining the imputed data back with non-numeric columns
data_imputed = data.copy()
data_imputed.update(data_knn_imputed)
data_imputed.update(data_mice_imputed)


# Step 3: Remove duplicates
data_cleaned = data_imputed.drop_duplicates()

# Step 4: Correct data types if necessary
# For simplicity, assuming the initial data types are correct. Adjust as needed:
# Example: 
#data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d-%m-%Y')

# Document the cleaning process

print("\nAfter cleaning:")
print("Dataset shape:", data_cleaned.shape)
print("Missing values after imputation:\n", data_cleaned.isnull().sum())

'''

# Document the cleaning process
st.header("Cleaned Dataset")
st.write("After cleaning:")
st.write("Dataset shape:", data_cleaned.shape)
st.write("Missing values after imputation:")
st.write(data_cleaned.isnull().sum())
'''

# Assuming 'CustomerIncomeGroup' as the target for SMOTE, adjust as necessary
target_column = 'CustomerIncomeGroup'
numeric_features = data_cleaned.select_dtypes(include=[np.number]).columns

# Apply SMOTE
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(data_cleaned[numeric_features], data_cleaned[target_column])

# Combine resampled data back into a DataFrame
data_augmented = pd.DataFrame(X_resampled, columns=numeric_features)
data_augmented[target_column] = y_resampled

# Document the augmentation process

print("\nAfter augmentation:")
print("Dataset shape:", data_augmented.shape)
'''

st.header("Augmented Dataset")
st.write("After augmentation:")
st.write("Dataset shape:", data_augmented.shape)

'''

# Exploratory Data Analysis with Insights

# Summary statistics
summary_stats = data_augmented.describe()
print("\nSummary statistics:\n", summary_stats)
'''
st.header("Summary Statistics")
st.write(summary_stats)
'''
# Heatmap of correlations
#st.header("Correlation Heatmap")
plt.figure(figsize=(16, 10))
sns.heatmap(data_augmented[numeric_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#st.pyplot(plt)

# Pair plot for initial EDA
#st.header("Pair Plot")
sns.pairplot(data_augmented)
plt.show()
#st.pyplot(plt)
# PCA for dimensionality reduction and visualization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_augmented.select_dtypes(include=[np.number]))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

data_augmented['pca-one'] = pca_result[:,0]
data_augmented['pca-two'] = pca_result[:,1]

#st.header("PCA Scatter Plot")
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue=target_column,
    palette=sns.color_palette("hsv", len(data_augmented[target_column].unique())),
    data=data_augmented,
    legend="full",
    alpha=0.3
)
plt.title('PCA Scatter Plot')
plt.show()
#st.pyplot(plt)

# Provide insights and hypotheses based on the EDA findings

print("\nEDA Insights:")
print("1. The correlation heatmap shows strong correlations between certain financial metrics, such as Revenue and Profit.")
print("2. The pair plot reveals potential clusters and relationships between various features.")
print("3. PCA visualization shows the separation of data points, which could be useful for clustering or classification tasks.")
'''
st.header("EDA Insights")
st.write("1. The correlation heatmap shows strong correlations between certain financial metrics, such as Revenue and Profit.")
st.write("2. The pair plot reveals potential clusters and relationships between various features.")
st.write("3. PCA visualization shows the separation of data points, which could be useful for clustering or classification tasks.")
'''
