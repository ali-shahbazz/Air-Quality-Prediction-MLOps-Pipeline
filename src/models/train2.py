# train2.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AirPollutionDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        X = self.data[index:index + self.sequence_length, :-1]
        y = self.data[index + self.sequence_length, -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Output layer
        out = self.fc(out[:, -1, :])
        return out.squeeze()

class LSTMTrainer:
    def __init__(self):
        self.data_path = Path("data") / "global_air_pollution.csv"
        self.models_path = Path("models")
        self.scaler_path = self.models_path / "scaler.joblib"
        self.sequence_length = 10  # Adjust as needed
        self.batch_size = 32
        self.epochs = 10  # Adjust as needed
        self.learning_rate = 0.001

        self.models_path.mkdir(parents=True, exist_ok=True)

        # Set MLflow tracking URI to a local directory
        mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))

    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv(self.data_path)
        df = df.dropna().reset_index(drop=True)
        return df

    def preprocess_data(self, df):
        """Preprocess data for LSTM model"""
        # Create a synthetic 'Time' column
        df['Time'] = df.index

        # Select features and target
        features = [
            "Time",
            "CO AQI Value",
            "Ozone AQI Value",
            "NO2 AQI Value",
            "PM2.5 AQI Value"
        ]
        target = "AQI Value"

        # Scale the data
        data = df[features + [target]].values
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")

        return data_scaled

    def train_model(self, data_scaled, params):
        """Train the LSTM model with given hyperparameters"""
        dataset = AirPollutionDataset(data_scaled, self.sequence_length)
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        input_size = data_scaled.shape[1] - 1  # Number of features
        model = LSTMModel(input_size, params['hidden_size'], params['num_layers'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Start MLflow run
        with mlflow.start_run():
            mlflow.log_params(params)

            model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                mlflow.log_metric('train_loss', avg_loss, step=epoch+1)

            # Validate the model
            model.eval()
            with torch.no_grad():
                y_preds = []
                y_trues = []
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    y_preds.extend(outputs.numpy())
                    y_trues.extend(y_batch.numpy())

                # Calculate metrics
                rmse = mean_squared_error(y_trues, y_preds, squared=False)
                mae = mean_absolute_error(y_trues, y_preds)
                r2 = r2_score(y_trues, y_preds)

                logger.info(f"Validation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('r2', r2)

            # Log the model
            # Provide an input example for signature
            input_example = torch.zeros(1, self.sequence_length, input_size)
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                input_example=input_example
            )

        return model, rmse

    def hyperparameter_tuning(self, data_scaled, num_trials=5):
        """Perform hyperparameter tuning using random search"""
        param_grid = {
            'hidden_size': [32, 50, 64, 100],
            'num_layers': [1, 2, 3]
        }

        best_rmse = float('inf')
        best_model = None
        best_params = None

        for _ in range(num_trials):
            params = {
                'hidden_size': random.choice(param_grid['hidden_size']),
                'num_layers': random.choice(param_grid['num_layers'])
            }
            logger.info(f"Training with params: {params}")
            model, rmse = self.train_model(data_scaled, params)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params

        # Save the best model
        model_path = self.models_path / f"best_lstm_model.pt"
        torch.save(best_model.state_dict(), model_path)
        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Best params: {best_params}, Best RMSE: {best_rmse:.4f}")

    def main(self):
        df = self.load_data()
        data_scaled = self.preprocess_data(df)
        self.hyperparameter_tuning(data_scaled)

if __name__ == "__main__":
    trainer = LSTMTrainer()
    trainer.main()