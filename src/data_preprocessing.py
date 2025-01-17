import yfinance as yf
import pandas as pd
from datasets import Dataset

class DataProcessor:
    def __init__(self):
        pass
    
    def load_stock_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data for a given ticker symbol and date range.
        """
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        return stock_data
    
    def load_twitter_data(self, dataset_name):
        """
        Load a dataset from the Hugging Face Hub using the 'datasets' library.
        """
        dataset = Dataset.load_dataset(dataset_name)
        return dataset['train']  # Assuming 'train' split contains the data
    
    def process_stock_data(self, stock_df):
        """
        Process stock data by converting the 'Date' column to datetime and handling missing values.
        """
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df.fillna(method='ffill', inplace=True)  # Forward fill missing values
        return stock_df
    
    def process_twitter_data(self, twitter_df):
        """
        Process Twitter data by selecting relevant columns and handling missing values.
        """
        twitter_df = twitter_df[['created_at', 'text']]
        twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'])
        twitter_df.fillna('', inplace=True)  # Replace NaN with empty string
        return twitter_df
