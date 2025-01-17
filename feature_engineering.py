import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_lag_features(self, df, lag=1):
      
        #Create lag features for time-series forecasting.
        df['Lag_Close'] = df['Close'].shift(lag)
        #Handling missing data
        return df.fillna(0) 
    
    def combine_sentiment_and_stock(self, stock_df, sentiment_df):

        #Combine sentiment scores with stock data for feature engineering.
        sentiment_df = sentiment_df.rename(columns={"created_at": "Date", "Sentiment": "Sentiment"})
        combined_df = pd.merge(stock_df, sentiment_df, on="Date", how="left")
        return combined_df
