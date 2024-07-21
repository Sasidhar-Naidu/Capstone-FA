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

# Net Income Analysis with Scenario Modeling
def scenario_modeling(df, scenario):
    if scenario == 'economic_downturn':
        df['NetIncome_Scenario'] = df['NetIncome'] * 0.8
    elif scenario == 'market_boom':
        df['NetIncome_Scenario'] = df['NetIncome'] * 1.2
    else:
        df['NetIncome_Scenario'] = df['NetIncome']
    return df

# Apply scenario modeling
data_base = scenario_modeling(data_cleaned.copy(), 'base')
data_downturn = scenario_modeling(data_cleaned.copy(), 'economic_downturn')
data_boom = scenario_modeling(data_cleaned.copy(), 'market_boom')

# Combine the data for plotting
data_scenarios = data_cleaned[['Company', 'NetIncome']].copy()
data_scenarios['NetIncome_Economic_Downturn'] = data_downturn['NetIncome_Scenario']
data_scenarios['NetIncome_Market_Boom'] = data_boom['NetIncome_Scenario']

# Rename the base scenario for clarity
data_scenarios.rename(columns={'NetIncome': 'NetIncome_Base'}, inplace=True)

# Melt the data for seaborn plotting
data_scenarios_melted = data_scenarios.melt(id_vars='Company', var_name='Scenario', value_name='NetIncome')

# Plot the scenarios
plt.figure(figsize=(14, 7))
sns.barplot(x='Company', y='NetIncome', hue='Scenario', data=data_scenarios_melted)
plt.title('Net Income under Different Scenarios')
plt.xlabel('Company')
plt.ylabel('Net Income')
plt.legend(title='Scenario')
plt.show()

# ROA Analysis with Benchmarking
data_cleaned['ROA'] = data_cleaned['Profit'] / data_cleaned['Assets']
industry_roa_benchmark = data_cleaned.groupby('Industry')['ROA'].mean()

# Plot ROA
plt.figure(figsize=(10, 6))
sns.barplot(x=industry_roa_benchmark.index, y=industry_roa_benchmark.values)
plt.title('ROA Benchmarked Against Industry Standards')
plt.xlabel('Industry')
plt.ylabel('ROA')
plt.show()

# EPS by Industry with Time Series Analysis
if 'MarketCap' in data_cleaned.columns and 'EarningsPerShare' in data_cleaned.columns:
    data_cleaned['SharesOutstanding'] = data_cleaned['MarketCap'] / data_cleaned['EarningsPerShare']
else:
    print("\nColumns 'MarketCap' or 'EarningsPerShare' are missing, SharesOutstanding calculation skipped.")

if 'SharesOutstanding' in data_cleaned.columns:
    data_cleaned['EPS'] = data_cleaned['NetIncome'] / data_cleaned['SharesOutstanding']
    industry_eps = data_cleaned.groupby('Industry')['EPS'].mean()

    # Plot EPS
    plt.figure(figsize=(10, 6))
    sns.barplot(x=industry_eps.index, y=industry_eps.values)
    plt.title('EPS Compared Across Industries')
    plt.xlabel('Industry')
    plt.ylabel('EPS')
    plt.show()
else:
    print("\nColumns 'MarketCap' or 'EarningsPerShare' are missing, EPS calculation skipped.")

# Revenue Ranking with Advanced Visualization
data_cleaned['RevenueRank'] = data_cleaned['Revenue'].rank(ascending=False)
plt.figure(figsize=(14, 7))
sns.barplot(x='RevenueRank', y='Revenue', data=data_cleaned)
plt.title('Revenue Ranking')
plt.show()
print("\nRevenue ranking completed and visualized.")

# CAGR Calculation with Sensitivity Analysis

def calculate_cagr(start_value, end_value, periods):
    return ((end_value / start_value)**(1/periods) - 1) * 100

# Ensure 'Year' column is available
if 'Date' in data_cleaned.columns:
    data_cleaned['Year'] = data_cleaned['Date'].dt.year

# Calculate CAGR for each company based on available years
cagr_values = {}

for company in data_cleaned['Company'].unique():
    company_data = data_cleaned[data_cleaned['Company'] == company]
    min_year = company_data['Year'].min()
    max_year = company_data['Year'].max()
    
    if min_year != max_year:
        start_value = company_data[company_data['Year'] == min_year]['Revenue'].values[0]
        end_value = company_data[company_data['Year'] == max_year]['Revenue'].values[0]
        periods = max_year - min_year
        cagr = calculate_cagr(start_value, end_value, periods)
        cagr_values[company] = cagr

# CAGR Calculation with Sensitivity Analysis

def calculate_cagr(start_value, end_value, periods):
    return ((end_value / start_value)**(1/periods) - 1) * 100

# Ensure 'Year' column is available
if 'Date' in data_cleaned.columns:
    data_cleaned['Year'] = data_cleaned['Date'].dt.year

# Verify revenue data and calculate CAGR for each company
cagr_values = {}
revenue_data = {}

for company in data_cleaned['Company'].unique():
    company_data = data_cleaned[data_cleaned['Company'] == company]
    min_year = company_data['Year'].min()
    max_year = company_data['Year'].max()
    
    if min_year != max_year:
        start_value = company_data[company_data['Year'] == min_year]['Revenue'].values[0]
        end_value = company_data[company_data['Year'] == max_year]['Revenue'].values[0]
        periods = max_year - min_year
        cagr = calculate_cagr(start_value, end_value, periods)
        cagr_values[company] = cagr
        revenue_data[company] = (min_year, start_value, max_year, end_value)

# Convert CAGR values to DataFrame
cagr_df = pd.DataFrame(list(cagr_values.items()), columns=['Company', 'CAGR'])
revenue_df = pd.DataFrame.from_dict(revenue_data, orient='index', columns=['Start_Year', 'Start_Revenue', 'End_Year', 'End_Revenue'])

# Display revenue data for verification
print("\nRevenue data used for CAGR calculation:\n", revenue_df)
print("\nCAGR values for each company:\n", cagr_df)

# Plot CAGR values
plt.figure(figsize=(14, 7))
sns.barplot(x='Company', y='CAGR', data=cagr_df)
plt.title('CAGR for Each Company')
plt.xlabel('Company')
plt.ylabel('CAGR (%)')
plt.xticks(rotation=45)
plt.show()

# Sensitivity analysis
def sensitivity_analysis(df, column, factor):
    df[f'{column}_Sensitivity'] = df[column] * factor
    return df

# Example: Perform sensitivity analysis on NetIncome
data_sensitivity = sensitivity_analysis(data_cleaned.copy(), 'NetIncome', 1.1)
print("\nSensitivity analysis completed.")

# Profit Distribution with Statistical Analysis
plt.figure(figsize=(14, 7))
box = sns.boxplot(x='Industry', y='Profit', data=data_cleaned)
strip = sns.stripplot(x='Industry', y='Profit', data=data_cleaned, color='red', size=5, jitter=True, dodge=True)

# Adding median values on the plot
medians = data_cleaned.groupby(['Industry'])['Profit'].median().values
median_labels = [f'{med:2.2f}' for med in medians]

for tick, label in zip(range(len(medians)), median_labels):
    box.text(tick, medians[tick] + 1000, label, 
             horizontalalignment='center', size='small', color='w', weight='semibold')

plt.title('Profit Distribution by Industry')
plt.show()
print("\nProfit distribution visualized.")

# Display descriptive statistics for profit by industry
profit_stats = data_cleaned.groupby('Industry')['Profit'].describe()
print("\nDescriptive statistics for profit by industry:\n", profit_stats)

# Statistical analysis
import scipy.stats as stats
anova_result = stats.f_oneway(*[group['Profit'].values for name, group in data_cleaned.groupby('Industry')])
print("\nANOVA result for profit distribution by industry:")
print(anova_result)
