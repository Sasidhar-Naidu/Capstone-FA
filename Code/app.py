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

# Load the dataset
file_path = r'Dataset/financialanalytics_Dataset.csv'
data = pd.read_csv(file_path)

# Data cleaning and imputation
knn_imputer = KNNImputer(n_neighbors=5)
data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
data_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)
data_imputed = data.copy()
data_imputed.update(data_knn_imputed)
data_imputed.update(data_mice_imputed)
data_cleaned = data_imputed.drop_duplicates()
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%Y-%m-%d')

# Calculate Financial Ratios
def calculate_financial_ratios(df):
    df['GrossMargin'] = (df['Revenue'] - df['Expenses']) / df['Revenue']
    df['NetProfitMargin'] = df['Profit'] / df['Revenue']
    df['ROI'] = df['Profit'] / (df['Assets'] + df['Liabilities'])
    df['CurrentRatio'] = df['Assets'] / df['Liabilities']
    df['DebtToEquity'] = df['Liabilities'] / df['Equity']
    return df

data_cleaned = calculate_financial_ratios(data_cleaned)

# Streamlit interface
st.title("Financial Analytics Dashboard")

# Display initial dataset information
if st.button('Show Initial Dataset Info'):
    st.write("Initial dataset shape:", data.shape)
    st.write("Initial missing values:", data.isnull().sum())

# Display cleaned dataset information
if st.button('Show Cleaned Dataset Info'):
    st.write("Dataset shape after cleaning:", data_cleaned.shape)
    st.write("Missing values after imputation:", data_cleaned.isnull().sum())

# Display Financial Ratios for Each Company
if st.button('Show Financial Ratios for Each Company'):
    st.write(data_cleaned[['Company', 'GrossMargin', 'NetProfitMargin', 'ROI', 'CurrentRatio', 'DebtToEquity']])

# Display Average Financial Ratios by Industry
if st.button('Show Average Financial Ratios by Industry'):
    industry_ratios = data_cleaned.groupby('Industry')[['GrossMargin', 'NetProfitMargin', 'ROI', 'CurrentRatio', 'DebtToEquity']].mean()
    st.write(industry_ratios)

# Heatmap of correlations
if st.button('Show Correlation Heatmap'):
    plt.figure(figsize=(16, 10))
    sns.heatmap(data_cleaned.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt.gcf())

# PCA Scatter Plot
if st.button('Show PCA Scatter Plot'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cleaned.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    data_cleaned['pca-one'] = pca_result[:,0]
    data_cleaned['pca-two'] = pca_result[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue='CustomerIncomeGroup',
        palette=sns.color_palette("hsv", len(data_cleaned['CustomerIncomeGroup'].unique())),
        data=data_cleaned,
        legend="full",
        alpha=0.3
    )
    plt.title('PCA Scatter Plot')
    st.pyplot(plt.gcf())

# Time Series Analysis
company = st.text_input('Enter a Company Name for Time Series Analysis:')
if st.button('Show Time Series Analysis') and company:
    df_company = data_cleaned[data_cleaned['Company'] == company]
    plt.figure(figsize=(14, 7))
    plt.plot(df_company['Date'], df_company['Revenue'], label='Revenue')
    plt.plot(df_company['Date'], df_company['Expenses'], label='Expenses')
    plt.plot(df_company['Date'], df_company['Profit'], label='Profit')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title(f'Time Series Analysis for {company}')
    plt.legend()
    st.pyplot(plt.gcf())
