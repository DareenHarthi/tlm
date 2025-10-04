import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BaselineModels:
    """Collection of baseline models for age prediction"""
    
    def __init__(self):
        self.models = {}
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Train Random Forest Regressor"""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_kmeans_regression(self, X_train, y_train, n_clusters=10, random_state=42):
        """Train K-means clustering with separate linear regression for each cluster"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        train_clusters = kmeans.fit_predict(X_train)
        
        regressors = {}
        for i in range(n_clusters):
            X_cluster = X_train[train_clusters == i]
            y_cluster = y_train[train_clusters == i]
            if len(X_cluster) > 0:
                regressor = LinearRegression()
                regressor.fit(X_cluster, y_cluster)
                regressors[i] = regressor
        
        model = {'kmeans': kmeans, 'regressors': regressors}
        self.models['kmeans_regression'] = model
        return model
    
    def train_linear_regression(self, X_train, y_train):
        """Train simple Linear Regression"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, input_features=192, lr=0.001, n_epochs=1000):
        """Train Neural Network (MLP)"""
        model = MLPRegressor(input_features=input_features)
        model.fit(X_train, y_train, lr=lr, n_epochs=n_epochs)
        self.models['neural_network'] = model
        return model
    
    def train_commonsense(self, X_train, y_train):
        """Train commonsense baseline (mean age prediction)"""
        mean_age = np.mean(y_train)
        self.models['commonsense'] = mean_age
        return mean_age
    
    def predict(self, model_name, X_test):
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        if model_name == 'kmeans_regression':
            return self._predict_kmeans(model, X_test)
        elif model_name == 'neural_network':
            return model.predict(X_test).detach().numpy()
        elif model_name == 'commonsense':
            return np.full(len(X_test), model)
        else:
            return model.predict(X_test)
    
    def _predict_kmeans(self, model, X_test):
        """Helper function for K-means regression prediction"""
        kmeans = model['kmeans']
        regressors = model['regressors']
        
        predictions = np.zeros(len(X_test))
        for idx, x in enumerate(X_test):
            cluster = kmeans.predict(x.reshape(1, -1))[0]
            if cluster in regressors:
                predictions[idx] = regressors[cluster].predict(x.reshape(1, -1))[0]
            else:
                # Fallback to mean prediction if cluster has no regressor
                predictions[idx] = np.mean(list(regressors.values())[0].predict(x.reshape(1, -1)))
        
        return predictions
    
    def evaluate(self, model_name, X_test, y_test):
        """Evaluate model and return MAE and RMSE"""
        predictions = self.predict(model_name, X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return mae, rmse
    
    def get_available_models(self):
        """Return list of trained models"""
        return list(self.models.keys())


class MLPRegressor(nn.Module):
    """Multi-layer Perceptron for regression"""
    
    def __init__(self, input_features=192, hidden_size=128, fit_intercept=True):
        super(MLPRegressor, self).__init__()
        self.fit_intercept = fit_intercept
        self.hidden = nn.Linear(input_features, hidden_size, bias=fit_intercept)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1, bias=fit_intercept)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x)
    
    def predict(self, X):
        """Make predictions on input data"""
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return self(X).flatten()
    
    def fit(self, X, y, lr=0.001, n_epochs=1000, batch_size=32):
        """Train the neural network"""
        self.train()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
        
        return self


def train_all_baselines(X_train, y_train):
    """Train all baseline models and return the collection"""
    baselines = BaselineModels()
    
    print("Training Commonsense (Mean)...")
    baselines.train_commonsense(X_train, y_train)
    
    print("Training Random Forest...")
    baselines.train_random_forest(X_train, y_train)
    
    print("Training K-means + Regression...")
    baselines.train_kmeans_regression(X_train, y_train)
    
    print("Training Linear Regression...")
    baselines.train_linear_regression(X_train, y_train)
    
    print("Training Neural Network...")
    baselines.train_neural_network(X_train, y_train)
    
    return baselines


def evaluate_all_baselines(baselines, X_test, y_test):
    """Evaluate all trained baseline models"""
    results = {}
    
    for model_name in baselines.get_available_models():
        mae, rmse = baselines.evaluate(model_name, X_test, y_test)
        results[model_name] = {'MAE': mae, 'RMSE': rmse}
    
    return results