#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:37:49 2024

@author: apple
"""
#%% import repertories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas_datareader.data as web
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.metrics import brier_score_loss, roc_curve, auc, log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
#%%1. DataCleaning
# read "mpd_stats.csv" with relative path
mpd_stats = pd.read_csv("/Users/apple/Downloads/科科/2023-IAQR-BU-MSMFT/data/mpd_stats.csv", skiprows=2, header = 1)
# convert the "Date" column to datetime
mpd_stats["idt"] = pd.to_datetime(mpd_stats["idt"])
#rename the idt column to Date
mpd_stats.rename(columns = {"idt":"Date"}, inplace = True)

mpd_stats.info()

# check for missing values
mpd_stats.isnull().sum()

# # Filter rows where 'Date' is between 2010 and 2023
# Filtering the dataset to include only the records from the year 2012 onwards
filtered_mpd_stats = mpd_stats[(mpd_stats['Date'].dt.year >= 2012)].copy()
filtered_mpd_stats.isnull().sum()

# Extract year from Date for easier analysis
filtered_mpd_stats['Year'] = filtered_mpd_stats['Date'].dt.year

# Identify markets that have data for each year in the range 2013 to 2023
years = range(2012, 2024)  # 2023 is inclusive
markets_with_complete_data = filtered_mpd_stats.groupby('market').filter(lambda x: all(y in x['Year'].values for y in years))['market'].unique()

# Filter the original dataframe to keep only the markets identified
mpd_stats_filtered_complete = mpd_stats[mpd_stats['market'].isin(markets_with_complete_data)]
mpd_stats_filtered_complete.isnull().sum()

# # remove rows that date is before 2013 jan 10
mpd_stats_filtered_complete = mpd_stats_filtered_complete[mpd_stats_filtered_complete['Date'] >= '2013-01-10']

print("Number of Unique market variables: ", len(mpd_stats_filtered_complete['market'].unique()))
print("Listed Market Variables: ", mpd_stats_filtered_complete['market'].unique())

# print the staring and ending date for each market in mpd_stats_filtered_complete
start_end_dates = mpd_stats_filtered_complete.groupby('market').agg(start_date=('Date', 'min'), end_date=('Date', 'max'))
print(start_end_dates)


#Upsampling for weekly data by forward filling, as majority of the market data frequency are in weekly basis
mpd_stats_filtered_complete.set_index('Date', inplace=True)

#2014-09-04之前的数据都是两周一次，然后用前向填充把缺失的数据填充上，填充后的数据是每周一次。
def resample_and_fill(group):
    # Define the cutoff date
    cutoff_date = '2014-09-04'
    
    # Filter the group for dates before the cutoff
    group_before_cutoff = group[:cutoff_date]
    
    # Resample to weekly frequency and forward fill
    group_resampled = group_before_cutoff.resample('W-Thu').ffill()
    
    # Concatenate with the part of the group after the cutoff date
    group_after_cutoff = group[pd.to_datetime(cutoff_date) + pd.Timedelta(days=1):]
    result = pd.concat([group_resampled, group_after_cutoff])
    
    return result

# Apply the function to each group and combine the results
mpd_stats_weekly = mpd_stats_filtered_complete.groupby('market', group_keys=False).apply(resample_and_fill)

# Reset the index
mpd_stats_weekly.reset_index(inplace=True)

mpd_stats_weekly.isnull().sum()
#把maturity_target用前向填充填充掉
mpd_stats_weekly.fillna(method='ffill', inplace=True)
mpd_stats_weekly.isnull().sum()

# export the cleaned data to a csv file
mpd_stats_weekly.to_csv('/Users/apple/Downloads/科科/2023-IAQR-BU-MSMFT/data/mpd_stats_cleaned.csv', index=True)


#%%2.Feature Engineering
mpd_stats_weekly['market'].unique()
mpd_stats_weekly.info()

# create a df for each market with column name of market
features = pd.DataFrame(mpd_stats_weekly['market'].unique(), columns=['market'])
# add a column for with the name of feature_id
features['feature_name'] = mpd_stats_weekly['market'].unique()
# rename the market column to market category
features.rename(columns = {"market":"market_category"}, inplace = True)

# if the feature_id has "bac" 'citi', then change market_category to "Bank"
features.loc[features['feature_name'].isin(['bac', 'citi']), 'market_category'] = 'Bank'
# if the feature_id has "silver", "corn", "soybns", "gold", 'iyr', 'wheat', then change market_category to "Commodity"
features.loc[features['feature_name'].isin(['silver', 'corn', 'soybns', 'gold', 'iyr', 'wheat']), 'market_category'] = 'Commodity'
# if the feature_id has 'yen', 'euro', 'pound', then change market_category to "Currency"
features.loc[features['feature_name'].isin(['yen', 'euro', 'pound']), 'market_category'] = 'Currency'
# if the feature_id has 'sp12m', 'sp6m', then change market_category to "Equity"
features.loc[features['feature_name'].isin(['sp12m', 'sp6m']), 'market_category'] = 'Equity'
# if the feature_id has 'infl1y',  'infl2y', 'infl5y', then change market_category to "Inflation"
features.loc[features['feature_name'].isin(['infl1y', 'infl2y', 'infl5y']), 'market_category'] = 'Inflation'
# if the feature_id has 'tr10yr', 'tr5yr', 'LR3y3m','LR5y3m' then change market_category to "Rates"
features.loc[features['feature_name'].isin(['tr10yr', 'tr5yr', 'LR3y3m','LR5y3m']), 'market_category'] = 'Rates'

# group the features by market_category
features = features.sort_values(by='feature_name')
# add id number for each feature
features['id'] = range(1, len(features) + 1)
features

# Create a dictionary to map market names to feature IDs
market_to_id = features.set_index('feature_name')['id'].to_dict()

# Initialize df_feature_engineered with the Date column
df_feature_engineered = pd.DataFrame(mpd_stats_weekly[mpd_stats_weekly['market'] == 'bac']['Date'].copy()).drop_duplicates()

# Iterate over each market to merge its data into df_feature_engineered
for market in mpd_stats_weekly['market'].unique():
    # Select the data for the current market, excluding the 'market' column
    market_data = mpd_stats_weekly[mpd_stats_weekly['market'] == market].drop(columns=['market']).copy()
    
    # Get the feature ID for the current market
    feature_id = market_to_id[market]
    
    # Rename the columns based on the feature ID, except for 'Date'
    market_data.rename(columns={col: f'f{feature_id}_{col}' for col in market_data.columns if col != 'Date'}, inplace=True)
    
    # Merge the data into df_feature_engineered
    df_feature_engineered = df_feature_engineered.merge(market_data, on='Date', how='left')

# This ensures that the 'market' column is not included in the final DataFrame
# df_feature_engineered就是把时间对齐了，然后列是不同资产的不同特征的值。

# find nan values or missing values
nan_rows = df_feature_engineered.isnull().sum()
#这个里面的缺失值应该是因为时间没有对齐。4，6，7，9，14

# Interpolate missing values linearly without using future data
df_feature_engineered['Date'] = pd.to_datetime(df_feature_engineered['Date'])
df_feature_engineered.set_index('Date', inplace=True)
df_feature_engineered.interpolate(method='time', inplace=True)

#这个cell可以不用，我们就接着cell（1）的mpd_stats_cleaned.csv做就可以。

#%%3.解决第一问，estimator是否有用
##（1）导入标普500的数据，并计算真实的return
sp500_data_path = '/Users/apple/Downloads/科科/2023-IAQR-BU-MSMFT/data/^SPX.csv'
sp500_data = pd.read_csv(sp500_data_path)

# Convert the 'Date' column to datetime format
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])

# Filter the data for the specified date range
start_date = '2013-01-10'
end_date = '2024-02-07'
sp500_filtered_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]

# Calculate the 6-month return
# First, ensure the data is sorted by date
sp500_filtered_data = sp500_filtered_data.sort_values('Date')

# Calculate the 6-month return. The shift will be based on trading days (~126 trading days in 6 months)
sp500_filtered_data['6m_return'] = sp500_filtered_data['Adj Close'].pct_change(periods=126).shift(-126)

#计算真实的return然后保存到一个数据框里
sp500_return = sp500_filtered_data[['Date', '6m_return']].dropna()

sp500_return['log_6m_return'] = np.log(1+sp500_return['6m_return'])

##（2）然后取出来sp6m和sp12m，因为methodology文件说已经全都整理成6个月后的预测了
mpd_stats_weekly.info
mpd_stats_weekly.isnull().sum()
#mpd_stats_weekly已经是一个整理好的没有缺失值的数据
# Filter the DataFrame for market 'sp6m'
df_sp6m = mpd_stats_weekly[mpd_stats_weekly['market'] == 'sp6m']

# Filter the DataFrame for market 'sp12m'
df_sp12m = mpd_stats_weekly[mpd_stats_weekly['market'] == 'sp12m']

##（3）比较真实的return和预测的均值

#根据df_sp6m中的日期，取出对应的真实return
# Merge to find matching dates and get the corresponding returns
merged_data_sp6m = pd.merge(df_sp6m, sp500_return, on='Date', how='left')
#根据df_sp12m中的日期，取出对应的真实return
merged_data_sp12m = pd.merge(df_sp12m, sp500_return, on='Date', how='left')

# Check if the 'Date' columns in both dataframes are exactly the same
dates_match = merged_data_sp6m['Date'].equals(merged_data_sp12m['Date'])
dates_match

# Renaming 'mu' columns to 'mu_sp6m' and 'mu_sp12m' respectively and merging.
sp_y = pd.merge(
    merged_data_sp6m[['Date', 'p50', 'p10', 'p90', '6m_return']],
    merged_data_sp12m[['Date', 'p50', 'p10', 'p90']],
    on='Date',
    suffixes=('_sp6m', '_sp12m')
)

# Drop any rows with NaN values to clean up the data.
sp_y.dropna(inplace=True)

# Plot all the selected columns in the sp_y dataframe.
plt.figure(figsize=(14, 7))

# Plotting the median (p50) as solid lines.
plt.plot(sp_y['Date'], sp_y['p50_sp6m'], label='Median 6m', color='Teal', linewidth=2)
plt.plot(sp_y['Date'], sp_y['p50_sp12m'], label='Median 12m', color='DarkOrange', linewidth=2)

# Plotting the actual return (6m_return) as a solid line.
plt.plot(sp_y['Date'], sp_y['6m_return'], label='Actual 6m Return', color='CornflowerBlue', linewidth=2)

# Plotting the 10th and 90th percentiles (p10, p90) as dashed lines.
plt.plot(sp_y['Date'], sp_y['p10_sp6m'], label='10th Percentile 6m', color='slategray', linestyle='--')
plt.plot(sp_y['Date'], sp_y['p90_sp6m'], label='90th Percentile 6m', color='slategray', linestyle='--')

# Filling the area between the 10th and 90th percentiles with a shaded region.
plt.fill_between(sp_y['Date'], sp_y['p10_sp6m'], sp_y['p90_sp6m'], color='slategray', alpha=0.1)

# Optionally, do the same for the 12-month predictions, if needed.
# plt.plot(sp_y['Date'], sp_y['p10_sp12m'], label='10th Percentile 12m', color='peachpuff', linestyle='--')
# plt.plot(sp_y['Date'], sp_y['p90_sp12m'], label='90th Percentile 12m', color='peachpuff', linestyle='--')
# plt.fill_between(sp_y['Date'], sp_y['p10_sp12m'], sp_y['p90_sp12m'], color='peachpuff', alpha=0.3)

# Setting the title and labels.
plt.title('S&P 500 Median Predicted vs Actual Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Values')

# Adding the legend outside the plot to not obscure data.
# Adding the legend outside the plot to not obscure data.
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()




##（4）通过置信区间比较差距
# create a function to calculate the accuracy of the predictions
def calculate_prediction_accuracy(df, actual_col, p10_col, p50_col, p90_col):
    # Count how many times the actual log return falls within the p10 and p90 interval
    within_interval = df.apply(lambda row: row[p10_col] <= row[actual_col] <= row[p90_col], axis=1).sum()
    # Count how many times the actual log return is greater than the median prediction
    above_median = (df[actual_col] > df[p50_col]).sum()
    # Calculate the total number of points
    total_points = df.shape[0]
    
    # Calculate the accuracy percentages
    interval_accuracy = within_interval / total_points
    median_accuracy = above_median / total_points
    
    return {
        'within_interval_accuracy': interval_accuracy,
        'above_median_accuracy': median_accuracy,
        'total_points': total_points
    }

# Apply the function to our merged data
# Assuming the actual log returns column in merged_data is 'log_6m_return'
accuracy_results_sp6m = calculate_prediction_accuracy(
    merged_data_sp6m, 
    actual_col='log_6m_return', 
    p10_col='p10', 
    p50_col='p50', 
    p90_col='p90'
)

print('the accuracy of sp6m predictions',accuracy_results_sp6m)
'''
{'within_interval_accuracy': 0.8615916955017301,
 'above_median_accuracy': 0.6591695501730104,
 'total_points': 578}
'''
accuracy_results_sp12m = calculate_prediction_accuracy(
    merged_data_sp12m, 
    actual_col='log_6m_return', 
    p10_col='p10', 
    p50_col='p50', 
    p90_col='p90'
)

print('the accuracy of sp12m predictions',accuracy_results_sp12m)
accuracy_results_sp12m

'''
{'within_interval_accuracy': 0.9446366782006921,
 'above_median_accuracy': 0.6003460207612457,
 'total_points': 578}
'''

#这是个80%的置信区间，预测应该是接近80%会比较好，这里都不够接近，虽然是被覆盖的数据更多了，
#可能有两方面的原因，首先是数据量并不够大，会有一定的偏差，另一个是预测的比较保守，就是预测的方差偏小。

##（5）看一下能不能通过方向性来预测是否准确
#先把真实数据的涨跌取出来
# Convert the 'log_6m_return' column to +1 if positive, -1 if negative
sp500_return['log_6m_return_sign'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x > 0 else -1)
sp500_return
#然后把预测的涨跌也算出来
# Function to calculate z-score adjusted for skewness and kurtosis using Cornish-Fisher expansion
def z_score_cf_expansion(q, skew, kurt):
    # z is the z-score for the standard normal distribution
    z = stats.norm.ppf(q)
    
    # Cornish-Fisher expansion
    z_adj = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * (kurt - 3) / 24 - (2 * z**3 - 5 * z) * (skew**2) / 36
    return z_adj

# Calculate the z-score for the 50th percentile (median)
q_median = 0.5
merged_data_sp6m['z_median'] = z_score_cf_expansion(q_median, merged_data_sp6m['skew'], merged_data_sp6m['kurt'])

# Calculate the adjusted probability of a positive return using the adjusted z-score for the 50th percentile
merged_data_sp6m['adj_prob_positive_return'] = 1 - stats.norm.cdf(merged_data_sp6m['z_median'])

# Display the new DataFrame with the calculated adjusted probabilities
merged_data_sp6m[['Date', 'mu', 'sd', 'skew', 'kurt', 'adj_prob_positive_return']]

###
merged_data_sp12m['z_median'] = z_score_cf_expansion(q_median, merged_data_sp12m['skew'], merged_data_sp12m['kurt'])

# Calculate the adjusted probability of a positive return using the adjusted z-score for the 50th percentile
merged_data_sp12m['adj_prob_positive_return'] = 1 - stats.norm.cdf(merged_data_sp12m['z_median'])

# Display the new DataFrame with the calculated adjusted probabilities
merged_data_sp12m[['Date', 'mu', 'sd', 'skew', 'kurt', 'adj_prob_positive_return']]

#通过brier_score, roc_auc, logloss来判断方向性预测的准确与否

# Convert actual sign of return to binary (1 for positive, 0 for non-positive)
sp500_return['actual_binary_outcome'] = sp500_return['log_6m_return_sign'].apply(lambda x: 1 if x == 1 else 0)

##sp6m
# Merge the actual outcomes with the predicted probabilities
eval_data_sp6m = pd.merge(merged_data_sp6m, sp500_return, on='Date')

# Calculate Brier score
brier_score_sp6m = brier_score_loss(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])

# Calculate ROC curve and AUC
fpr_sp6m, tpr_sp6m, thresholds_sp6m = roc_curve(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])
roc_auc_sp6m = auc(fpr_sp6m, tpr_sp6m)

# Calculate Log Loss
logloss_sp6m = log_loss(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])

# Print the evaluation metrics
(brier_score_sp6m, roc_auc_sp6m, logloss_sp6m)
#(0.30298877039810873, 0.6980581576893052, 0.8002290079689729)

'''

1.  **Brier Score**: This score ranges from 0 to 1, where 0 represents a perfect model and 1 represents the worst model.  A Brier score of 0.3030 indicates that, on average, the squared difference between the predicted probabilities and the actual outcomes is moderately high.  This suggests that the probabilities are not as close to the true outcomes as they could be, indicating room for improvement in the model.

2.  **ROC AUC**: The area under the ROC curve (AUC) is a measure of the model's ability to correctly classify the binary outcomes.  It ranges from 0.5 (no better than random chance) to 1 (perfect classification).  An AUC of 0.6981 indicates that the model has a moderate ability to discriminate between positive and non-positive returns.  This is better than random chance but not close to perfect.

3.  **Log Loss**: Also known as logistic loss or cross-entropy loss, this metric measures the performance of a classification model where the prediction is a probability between 0 and 1.  The log loss increases as the predicted probability diverges from the actual label.  A log loss of 0.8002 is relatively high, suggesting that the model's predicted probabilities could be more accurate.

In summary, while the model shows some predictive ability (as indicated by an AUC greater than 0.5), it is not highly accurate in predicting the sign of the returns, and there is a considerable probability error (as indicated by the Brier score and Log Loss).  The model may benefit from calibration to improve probability estimates or from incorporating additional features or different modeling techniques to enhance its predictive power.
'''

##sp12m
# Merge the actual outcomes with the predicted probabilities
eval_data_sp12m = pd.merge(merged_data_sp12m, sp500_return, on='Date')

# Calculate Brier score
brier_score_sp12m = brier_score_loss(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])

# Calculate ROC curve and AUC
fpr_sp12m, tpr_sp12m, thresholds_sp12m = roc_curve(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])
roc_auc_sp12m = auc(fpr_sp12m, tpr_sp12m)

# Calculate Log Loss
logloss_sp12m = log_loss(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])

# Print the evaluation metrics
(brier_score_sp12m, roc_auc_sp12m, logloss_sp12m)
# (0.30458583771273107, 0.716120218579235, 0.8034942614886871)

##（6）判断大幅涨跌预测的是否准确
#大幅度也类似正return负return这个问题，去做判断。
#就是说我先去算，真实的产生大幅度增减的数据有多少，有就是1，没有就是0。
# 定义大幅变化的阈值为20%
large_change_threshold = 0.20

# 计算大幅上涨和下跌
sp500_return['large_increase'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x >= large_change_threshold else 0)
sp500_return['large_decrease'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x <= -large_change_threshold else 0)

# 输出结果查看
print(sp500_return.head())
sp500_return.describe()

#然后我已经有那个涨跌的预测了，我再去和这个涨跌的预测去比较。也是用brier_score, roc_auc, logloss来判断预测的准确性。
#我应该有四组数据：6m涨，6m跌；12m涨，12m跌
#6m涨
eval_data_sp6m = pd.merge(merged_data_sp6m, sp500_return, on='Date')
# Calculate Brier score
brier_score_sp6m_in = brier_score_loss(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])

# Calculate ROC curve and AUC
fpr_sp6m_in, tpr_sp6m_in, thresholds_sp6m_in = roc_curve(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])
roc_auc_sp6m_in = auc(fpr_sp6m_in, tpr_sp6m_in)

# Calculate Log Loss
logloss_sp6m_in = log_loss(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])

# Print the evaluation metrics
(brier_score_sp6m_in, roc_auc_sp6m_in, logloss_sp6m_in)
#(0.012429016340377705, 0.9888475836431226, 0.05116827792489753)

#6m跌
# Calculate Brier score
brier_score_sp6m_de = brier_score_loss(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])

# Calculate ROC curve and AUC
fpr_sp6m_de, tpr_sp6m_de, thresholds_sp6m_de = roc_curve(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])
roc_auc_sp6m_de = auc(fpr_sp6m_de, tpr_sp6m_de)

# Calculate Log Loss
logloss_sp6m_de = log_loss(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])

# Print the evaluation metrics
(brier_score_sp6m_de, roc_auc_sp6m_de, logloss_sp6m_de)
#(0.013842216166729912, 0.6867158671586716, 0.09762668168530149)

#12m涨
eval_data_sp12m = pd.merge(merged_data_sp12m, sp500_return, on='Date')
# Calculate Brier score
brier_score_sp12m_in = brier_score_loss(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])

# Calculate ROC curve and AUC
fpr_sp12m_in, tpr_sp12m_in, thresholds_sp12m_in = roc_curve(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])
roc_auc_sp12m_in = auc(fpr_sp12m_in, tpr_sp12m_in)

# Calculate Log Loss
logloss_sp12m_in = log_loss(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])

# Print the evaluation metrics
(brier_score_sp12m_in, roc_auc_sp12m_in, logloss_sp12m_in)
#(0.019231624611220105, 0.9560099132589839, 0.11040701467531214)

#12m跌
# Calculate Brier score
brier_score_sp12m_de = brier_score_loss(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])

# Calculate ROC curve and AUC
fpr_sp12m_de, tpr_sp12m_de, thresholds_sp12m_de = roc_curve(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])
roc_auc_sp12m_de = auc(fpr_sp12m_de, tpr_sp12m_de)

# Calculate Log Loss
logloss_sp12m_de = log_loss(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])

# Print the evaluation metrics
(brier_score_sp12m_de, roc_auc_sp12m_de, logloss_sp12m_de)
#(0.023975969887039705, 0.6841328413284133, 0.1547707960291045)
#预测的上升很准，下降就不如上升准。

# Calculate ROC curves and AUCs for all cases
fpr_sp6m_in, tpr_sp6m_in, _ = roc_curve(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])
roc_auc_sp6m_in = auc(fpr_sp6m_in, tpr_sp6m_in)

fpr_sp6m_de, tpr_sp6m_de, _ = roc_curve(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])
roc_auc_sp6m_de = auc(fpr_sp6m_de, tpr_sp6m_de)

fpr_sp12m_in, tpr_sp12m_in, _ = roc_curve(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])
roc_auc_sp12m_in = auc(fpr_sp12m_in, tpr_sp12m_in)

fpr_sp12m_de, tpr_sp12m_de, _ = roc_curve(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])
roc_auc_sp12m_de = auc(fpr_sp12m_de, tpr_sp12m_de)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 6m increase
axs[0, 0].plot(fpr_sp6m_in, tpr_sp6m_in, color='blue', label=f'ROC curve (area = {roc_auc_sp6m_in:.2f})')
axs[0, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 0].set_xlim([0.0, 1.0])
axs[0, 0].set_ylim([0.0, 1.05])
axs[0, 0].set_xlabel('False Positive Rate')
axs[0, 0].set_ylabel('True Positive Rate')
axs[0, 0].set_title('6m Increase ROC')
axs[0, 0].legend(loc="lower right")

# 6m decrease
axs[0, 1].plot(fpr_sp6m_de, tpr_sp6m_de, color='blue', label=f'ROC curve (area = {roc_auc_sp6m_de:.2f})')
axs[0, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 1].set_xlim([0.0, 1.0])
axs[0, 1].set_ylim([0.0, 1.05])
axs[0, 1].set_xlabel('False Positive Rate')
axs[0, 1].set_ylabel('True Positive Rate')
axs[0, 1].set_title('6m Decrease ROC')
axs[0, 1].legend(loc="lower right")

# 12m increase
axs[1, 0].plot(fpr_sp12m_in, tpr_sp12m_in, color='green', label=f'ROC curve (area = {roc_auc_sp12m_in:.2f})')
axs[1, 0].plot([0, 1], [0, 1], color='darkgreen', linestyle='--')
axs[1, 0].set_xlim([0.0, 1.0])
axs[1, 0].set_ylim([0.0, 1.05])
axs[1, 0].set_xlabel('False Positive Rate')
axs[1, 0].set_ylabel('True Positive Rate')
axs[1, 0].set_title('12m Increase ROC')
axs[1, 0].legend(loc="lower right")

#12m decrease
axs[1, 1].plot(fpr_sp12m_de, tpr_sp12m_de, color='green', label=f'ROC curve (area = {roc_auc_sp12m_de:.2f})')
axs[1, 1].plot([0, 1], [0, 1], color='darkgreen', linestyle='--')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('12m Decrease ROC')
axs[1, 1].legend(loc="lower right")

#Adjust layout
plt.tight_layout()
plt.show()


#%%4.解决第二问，to predict future returns, volatilities, reversals。
#（1）predict future returns
#就是真实的log return作为y，然后处理数据，log也要做，然后交叉项也要做，然后各个特征都要扔进来，我来跑回归

# 预处理数据
# 选择自变量，排除'lg_change_decr'和'lg_change_incr'
features = ['mu', 'sd', 'skew', 'kurt', 'p10', 'p50', 'p90']
X = df_sp6m[features]

# 生成交叉项
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
X_poly_df['Date'] = df_sp6m['Date'].values
X_poly_df.set_index('Date', inplace=True)

# 划分时间段
df = sp500_return[['Date', 'log_6m_return']]
df.set_index('Date', inplace=True)

# 计算80%的数据时间点
eighty_percent_index = int(len(df) * 0.8)
eighty_percent_date = df.iloc[eighty_percent_index]['Date']
eighty_percent_date

split_date = pd.Timestamp('2021-06-25')
train = df.loc[df['Date'] <= split_date]
test = df.loc[df['Date'] > split_date]


# 训练模型
X_train = X_poly_df.loc[df['Date'] <= split_date]

y_train = train['log_6m_return']
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# 模型回测
X_test = test[poly.get_feature_names_out(X.columns)]
y_test = test['log_6m_return']
predictions = model.predict(sm.add_constant(X_test))

# 评估模型
mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')
print(model.summary())




#（2）搞volatilities
#这里是基于真实的波动的分析
'''
要计算每天的真实波动率，你需要首先决定你想度量的波动率的时间范围。
通常，波动率是通过计算一定时间窗口内的收益率变动来估计的。
对于每日波动率的计算，可以选择一个固定的时间窗口（比如过去5天、10天或20天）
来计算这个窗口内的日收益率的标准差。这种方法通常被称为滚动波动率计算。
我先选择5天为窗口。
'''
#那我就是要先滚动的把真实的波动性算出来

# 计算窗口期为5天的对数回报率的标准差，即波动率
# 使用rolling()函数来创建一个滑动窗口，然后使用std()函数计算每个窗口期的标准差
sp500_return['5d_rolling_volatility'] = sp500_return['log_6m_return'].rolling(window=5).std()

# 查看结果
print(sp500_return[['Date', 'log_6m_return', '5d_rolling_volatility']])
sp500_return.dropna()


#然后去和预测的标准差比
data = pd.merge(sp500_return, merged_data_sp6m[['Date', 'sd']], on='Date', how='left', suffixes=('', '_sp6m'))
data = pd.merge(data, merged_data_sp12m[['Date', 'sd']], on='Date', how='left', suffixes=('', '_sp12m'))
data.dropna(inplace=True)

# 绘制时序图
plt.figure(figsize=(12, 6))

# 真实的5天滚动波动率
plt.plot(data['Date'], data['5d_rolling_volatility'], label='Real 5-Day Rolling Volatility', color='blue')

# 预测的波动率 - sp6m
plt.plot(data['Date'], data['sd'], label='Predicted Volatility (6m)', color='green')

# 预测的波动率 - sp12m
plt.plot(data['Date'], data['sd_sp12m'], label='Predicted Volatility (12m)', color='red')

plt.title('S&P 500 Real vs Predicted Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

#中心化之后再画一个图，看看是不是像的
# 计算需要中心化的列的均值
mean_5d_rolling_volatility = data['5d_rolling_volatility'].mean()
mean_sd = data['sd'].mean()
mean_sd_sp12m = data['sd_sp12m'].mean()

# 从每个列中减去其均值进行中心化
data['5d_rolling_volatility_centered'] = data['5d_rolling_volatility'] - mean_5d_rolling_volatility
data['sd_centered'] = data['sd'] - mean_sd
data['sd_sp12m_centered'] = data['sd_sp12m'] - mean_sd_sp12m


# 绘制时序图
plt.figure(figsize=(12, 6))

# 真实的5天滚动波动率
plt.plot(data['Date'], data['5d_rolling_volatility_centered'], label='Real 5-Day Rolling Volatility_centered', color='blue')

# 预测的波动率 - sp6m
plt.plot(data['Date'], data['sd_centered'], label='Predicted Volatility (6m)_centered', color='green')

# 预测的波动率 - sp12m
plt.plot(data['Date'], data['sd_sp12m_centered'], label='Predicted Volatility (12m)_centered', color='red')

plt.title('S&P 500 Real vs Predicted Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

#这两个预测是非常像的，预测出的走势基本类似。但是对于真实的波动的预测，感觉它只能预测出什么时候会发生较大的波动。

#标准之后再画一个图，看看是不是像的
# 创建一个StandardScaler对象
scaler = StandardScaler()

# 选择需要标准化的列
columns_to_scale = ['5d_rolling_volatility', 'sd', 'sd_sp12m']

# 对选定的列进行标准化
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# 绘制时序图
plt.figure(figsize=(12, 6))

# 真实的5天滚动波动率
plt.plot(data['Date'], data['5d_rolling_volatility'], label='Real 5-Day Rolling Volatility', color='blue')

# 预测的波动率 - sp6m
plt.plot(data['Date'], data['sd'], label='Predicted Volatility (6m)', color='green')

# 预测的波动率 - sp12m
plt.plot(data['Date'], data['sd_sp12m'], label='Predicted Volatility (12m)', color='red')

plt.title('S&P 500 Real vs Predicted Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

#标准化之后的数据趋势相似性很明显。可以认为是一个对波动比较好的预测。

#现在我基于VIX分析
#先画图分析预测的准确性
#然后基本就是（1）中的方法，只不过这次的y变成VIX的值，然后我做一个5日平均，去和已经预测出来的数据对齐


merged_data_sp6m

merged_data_sp12m


#（3）然后搞reversals
#reversals就是价格的反转。
#现在的思路是，我去把真实的这个反转的表征表示出来，然后把它当作y，去做一个回归，看这些估计量对预测有没有帮助
#因为目前看比较可能和reversals相关的因素有偏度，峰度，波动率。
'''
#1. 分析概率分布的形状
#偏度（Skewness）：概率分布的偏度可以提供市场对未来价格变动方向的偏好信息。正偏度表明市场预期更多的上行风险（即价格上涨的可能性较大），而负偏度则表明更多的下行风险（即价格下跌的可能性较大）。
#峰度（Kurtosis）：概率分布的峰度反映了极端价格变动的可能性。高峰度表明极端价格变动的概率较高，这可能是潜在反转的信号。
#2. 识别隐含波动率的变化
#期权市场的隐含波动率提供了市场对未来价格不确定性的度量。波动率的显著变化，特别是波动率微笑或波动率曲面的形状变化，可以指示市场情绪的变化，可能预示着价格反转。
'''

'''
可能能够作为反转的量化指标的变量
1. 变化率与方向
价格变化率（Rate of Change, ROC）：计算特定时间窗口内价格的变化率。您可以设定一个阈值，例如，如果价格变化率超过某个特定的正值或负值，可以认为发生了反转。
2. 相对强弱指数（RSI）
RSI 级别的变化：RSI 是衡量过去价格变动速度和变化的指标，值范围从0到100。您可以使用RSI的显著变化来量化反转的程度，如从超买区域（>70）跌至正常区域，或从超卖区域（<30）升至正常区域。
3. 移动平均线
移动平均线交叉：短期移动平均线（如5日或10日）与长期移动平均线（如50日或200日）的交叉，可以标记为上升反转（金叉）或下降反转（死叉）的信号。
4. 布林带宽度
布林带宽度变化：布林带宽度的增加可能表明市场波动性增加，与可能的价格反转相关。您可以通过观察布林带宽度的变化来评估反转的潜在强度。
5. 高低差价（High-Low Range）
高低差价：观察一定时间内的最高价与最低价之差，大的高低差价可能表明市场在该期间内经历了重要的反转。
'''


#%%解决第三问，如何基于已知信息做策略。这个和第四问一起做。

#思路一：我有了方差，我去找相关资产把波动对冲到零

#思路二：我能够预测reversals，然后知道价格立即下跌时提前抛售，知道价格立即上升时多多买入。

#这个策略运行之后最好要去回测一下。
#怎么回测也要想一下。回测出结果要做一张大图。属于很关键的信息。






















