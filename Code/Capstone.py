import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
file_path = r'Dataset\financialanalytics_Dataset.csv'
data = pd.read_csv(file_path)
print("Initial dataset shape:", data.shape)
print("Initial missing values:\n", data.isnull().sum())

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
numeric_data = data.select_dtypes(include=[np.number])
data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Multivariate Imputation
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
data_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Combining the imputed data back with non-numeric columns
data_imputed = data.copy()
data_imputed[numeric_data.columns] = data_mice_imputed

# Remove duplicates
data_cleaned = data_imputed.drop_duplicates()

# Correct data types if necessary
if 'Date' in data_cleaned.columns:
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%Y-%m-%d')
print("\nAfter cleaning:")
print("Dataset shape:", data_cleaned.shape)
print("Missing values after imputation:\n", data_cleaned.isnull().sum())

# Calculate Financial Ratios
def calculate_financial_ratios(df):
    df['GrossMargin'] = (df['Revenue'] - df['Expenses']) / df['Revenue']
    df['NetProfitMargin'] = df['Profit'] / df['Revenue']
    df['ROI'] = df['Profit'] / (df['Assets'] + df['Liabilities'])
    df['CurrentRatio'] = df['Assets'] / df['Liabilities']
    df['DebtToEquity'] = df['Liabilities'] / df['Equity']
    return df

# Apply financial ratios calculation
data_cleaned = calculate_financial_ratios(data_cleaned)

# Print Financial Ratios for Each Company
print("\nFinancial Ratios for Each Company:")
print(data_cleaned[['Company', 'GrossMargin', 'NetProfitMargin', 'ROI', 'CurrentRatio', 'DebtToEquity']])

# Print Financial Ratios by Industry
def print_industry_ratios(df):
    industry_ratios = df.groupby('Industry')[['GrossMargin', 'NetProfitMargin', 'ROI', 'CurrentRatio', 'DebtToEquity']].mean()
    print("\nAverage Financial Ratios by Industry:")
    print(industry_ratios)

print_industry_ratios(data_cleaned)

# Exploratory Data Analysis with Insights
# Summary statistics
summary_stats = data_cleaned.describe()
print("\nSummary statistics:\n", summary_stats)

# Heatmap of correlations
numeric_features = data_cleaned.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(16, 10))
sns.heatmap(data_cleaned[numeric_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# PCA for dimensionality reduction and visualization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned[numeric_features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
data_cleaned['pca-one'] = pca_result[:,0]
data_cleaned['pca-two'] = pca_result[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='Industry',  # Assuming 'Industry' is the target column
    palette=sns.color_palette("hsv", len(data_cleaned['Industry'].unique())),
    data=data_cleaned,
    legend="full",
    alpha=0.3
)
plt.title('PCA Scatter Plot')
plt.show()

print("\nEDA Insights:")
print("1. The correlation heatmap shows strong correlations between certain financial metrics, such as Revenue and Profit.")
print("2. PCA visualization shows the separation of data points, which could be useful for clustering or classification tasks.")

# Time Series Analysis
def plot_time_series(df, company):
    df_company = df[df['Company'] == company]
    plt.figure(figsize=(14, 7))
    plt.plot(df_company['Date'], df_company['Revenue'], label='Revenue')
    plt.plot(df_company['Date'], df_company['Expenses'], label='Expenses')
    plt.plot(df_company['Date'], df_company['Profit'], label='Profit')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title(f'Time Series Analysis for {company}')
    plt.legend()
    plt.show()

# Example: Plotting time series for Company A
plot_time_series(data_cleaned, 'A')

# Industry Comparison
def compare_industries(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    industry_means = df.groupby('Industry')[numeric_columns].mean()
    industry_means[['Revenue', 'Expenses', 'Profit']].plot(kind='bar', figsize=(14, 7))
    plt.title('Industry Comparison')
    plt.ylabel('Mean Values')
    plt.show()

compare_industries(data_cleaned)

# Beta and Volatility Analysis
def plot_beta_volatility(df):
    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=df, x='Beta', y='Volatility', hue='Industry', size='MarketCap', sizes=(20, 200))
    plt.title('Beta and Volatility Analysis')
    plt.xlabel('Beta')
    plt.ylabel('Volatility')
    plt.show()

plot_beta_volatility(data_cleaned)

# Customer Analysis
def plot_customer_analysis(df):
    customer_groups = df.groupby(['CustomerAgeGroup', 'CustomerIncomeGroup']).mean(numeric_only=True)
    customer_groups['NetProfitMargin'].unstack().plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Customer Age and Income Group vs Net Profit Margin')
    plt.ylabel('Net Profit Margin')
    plt.show()

plot_customer_analysis(data_cleaned)

# Geographic Analysis
def plot_geographic_analysis(df):
    geo_groups = df.groupby('GeographicLocation').mean(numeric_only=True)
    geo_groups[['Revenue', 'Profit', 'MarketCap']].plot(kind='bar', figsize=(14, 7))
    plt.title('Geographic Location Analysis')
    plt.ylabel('Mean Values')
    plt.show()

plot_geographic_analysis(data_cleaned)

# Save the updated dataset with financial ratios
data_cleaned.to_csv('financial_analysis_with_ratios.csv', index=False)

print("Analysis completed and results saved to 'financial_analysis_with_ratios.csv'")
