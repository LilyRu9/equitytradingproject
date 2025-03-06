import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load datasets from saved CSVs
matching_dates = pd.read_csv('matching_dates.csv', parse_dates=['Date'])
non_matching_dates = pd.read_csv('non_matching_dates.csv', parse_dates=['Date'])

# Compute daily returns
matching_dates['return'] = matching_dates.groupby('Ticker')['Close'].pct_change()
non_matching_dates['return'] = non_matching_dates.groupby('Ticker')['Close'].pct_change()

# Compute volatility (rolling standard deviation of returns over a 5-day window)
matching_dates['volatility'] = matching_dates.groupby('Ticker')['return'].rolling(window=5).std().reset_index(0, drop=True)
non_matching_dates['volatility'] = non_matching_dates.groupby('Ticker')['return'].rolling(window=5).std().reset_index(0, drop=True)

# Extract returns and volatility
news_returns = matching_dates['return']
no_news_returns = non_matching_dates['return']
news_volatility = matching_dates['volatility']
no_news_volatility = non_matching_dates['volatility']

# Perform statistical tests
return_t_stat, return_p_value = ttest_ind(news_returns.dropna(), no_news_returns.dropna(), equal_var=False)
volatility_t_stat, volatility_p_value = ttest_ind(news_volatility.dropna(), no_news_volatility.dropna(), equal_var=False)

# Print results
print()
print(f"Return T-stat: {return_t_stat}, P-value: {return_p_value}")
print(f"Volatility T-stat: {volatility_t_stat}, P-value: {volatility_p_value}")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Mean Return (News)', 'Mean Return (No News)', 'Mean Volatility (News)', 'Mean Volatility (No News)'],
    'Value': [news_returns.mean(), no_news_returns.mean(), news_volatility.mean(), no_news_volatility.mean()]
})
summary_stats.to_csv('baseline_summary.csv', index=False)

# Save processed datasets
matching_dates.to_csv('processed_matching_dates.csv', index=False)
non_matching_dates.to_csv('processed_non_matching_dates.csv', index=False)

# Adjust histogram range based on percentiles to avoid extreme outliers
return_lower, return_upper = np.percentile(news_returns.dropna().tolist() + no_news_returns.dropna().tolist(), [1, 99])
volatility_lower, volatility_upper = np.percentile(news_volatility.dropna().tolist() + no_news_volatility.dropna().tolist(), [1, 99])

# Plot histograms of returns
plt.figure(figsize=(12,5))
plt.hist(no_news_returns.dropna(), bins=50, alpha=0.5, label='No News Days', color='red', range=(return_lower, return_upper))
plt.hist(news_returns.dropna(), bins=50, alpha=0.5, label='News Days', color='blue', range=(return_lower, return_upper))
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title(f'Distribution of Daily Returns on News vs. Non-News Days\nT-stat: {return_t_stat:.2f}, P-value: {return_p_value:.2e}')
plt.legend()
plt.savefig('daily_returns_distribution.png')
plt.show()

# Plot histograms of volatility
plt.figure(figsize=(12,5))
plt.hist(no_news_volatility.dropna(), bins=50, alpha=0.5, label='No News Days', color='red', range=(volatility_lower, volatility_upper))
plt.hist(news_volatility.dropna(), bins=50, alpha=0.5, label='News Days', color='blue', range=(volatility_lower, volatility_upper))
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.title(f'Distribution of Volatility on News vs. Non-News Days\nT-stat: {volatility_t_stat:.2f}, P-value: {volatility_p_value:.2e}')
plt.legend()
plt.savefig('volatility_distribution.png')
plt.show()

# Plot histogram of returns for News Days only
plt.figure(figsize=(12,5))
plt.hist(news_returns.dropna(), bins=50, alpha=0.7, color='blue', range=(return_lower, return_upper))
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title(f'Distribution of Daily Returns on News Days')
plt.savefig('news_days_returns.png')
plt.show()

# Plot histogram of volatility for News Days only
plt.figure(figsize=(12,5))
plt.hist(news_volatility.dropna(), bins=50, alpha=0.7, color='blue', range=(volatility_lower, volatility_upper))
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.title(f'Distribution of Volatility on News Days')
plt.savefig('news_days_volatility.png')
plt.show()

