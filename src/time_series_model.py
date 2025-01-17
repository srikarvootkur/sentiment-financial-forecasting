import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_lightning import Trainer

class TimeSeriesForecaster:
    def __init__(self, input_chunk_length=60, output_chunk_length=30, batch_size=64, hidden_size=64, max_epochs=10):
        """
        Initialize the time series forecaster with the specified parameters.
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs

    def create_time_series(self, data):
        """
        Prepare the time-series data for the TFT model.
        """
        data['Date'] = pd.to_datetime(data['Date'])
        training = TimeSeriesDataSet(
            data,
            time_idx="Date",
            target="Close",  # Stock Price
            group_ids=["Stock"],  # Group for time-series data, e.g., each stock
            static_categoricals=["Stock"],  # Static features like stock name
            time_varying_known_reals=["Sentiment"],  # External variables like sentiment
            time_varying_unknown_reals=["Close"],  # Variables that are not available during prediction
            max_encoder_length=self.input_chunk_length,
            max_prediction_length=self.output_chunk_length,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        return training

    def train_model(self, training_data):
        """
        Train the TFT model on the prepared data.
        """
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        model = TemporalFusionTransformer.from_dataset(training_data, hidden_size=self.hidden_size, dropout=0.1)
        trainer = Trainer(max_epochs=self.max_epochs, gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(model, train_dataloader)
        return model

    def forecast(self, model, data, forecast_horizon=30):
        """
        Generate predictions using the trained model.
        """
        # Prepare the data for prediction
        data['Date'] = pd.to_datetime(data['Date'])
        prediction_data = TimeSeriesDataSet.from_dataset(
            model,
            data,
            predict=True,
            stop_randomization=True
        )
        # Create a DataLoader for the prediction data
        prediction_dataloader = torch.utils.data.DataLoader(prediction_data, batch_size=self.batch_size, shuffle=False)
        # Generate predictions
        predictions = model.predict(prediction_dataloader, n=forecast_horizon)
        return predictions
