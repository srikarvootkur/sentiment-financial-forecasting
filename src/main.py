from datetime import datetime
from data_preprocessing import DataProcessor
from sentiment_analysis import SentimentAnalyzer
from time_series_model import TimeSeriesForecaster
from feature_engineering import FeatureEngineer
from utils import Utils

def main():
    # Initialize necessary classes
    data_processor = DataProcessor()
    sentiment_analyzer = SentimentAnalyzer()
    forecaster = TimeSeriesForecaster(input_chunk_length=60, output_chunk_length=30)
    feature_engineer = FeatureEngineer()
    
    # Load stock data (from Yahoo Finance or local CSV)
    stock_data = data_processor.load_stock_data('AAPL', start_date="2015-01-01", end_date="2021-01-01")
    stock_data = data_processor.process_stock_data(stock_data)
    
    # Load and process Twitter data
    twitter_data = data_processor.load_twitter_data("financial_sentiment_dataset")
    sentiment_scores = sentiment_analyzer.batch_analyze_sentiment(twitter_data['text'])
    
    # Feature engineering: combine stock data with sentiment
    stock_data['Sentiment'] = sentiment_scores
    feature_data = feature_engineer.create_lag_features(stock_data)
    
    # Create and train the TFT model
    training_data = forecaster.create_time_series(feature_data)
    model = forecaster.train_model(training_data)
    
    # Make predictions
    forecast = forecaster.forecast(model, feature_data, forecast_horizon=30)

    # Save or visualize results
    Utils.save_to_csv(forecast, 'forecasted_stock_prices.csv')
    print(forecast.head())

if __name__ == "__main__":
    main()
