#!/usr/bin/env python3

import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SilentReimbursementMLModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_once()
        
    def _train_once(self):
        """Train the model once on initialization"""
        X, y = self.load_data()
        self.train_models(X, y)
        self.create_specialized_models(X, y)
        
    def load_data(self, filepath='public_cases.json'):
        """Load and prepare training data from public cases"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        features = []
        targets = []
        
        for case in data:
            inp = case['input']
            days = inp['trip_duration_days']
            miles = inp['miles_traveled']
            receipts = inp['total_receipts_amount']
            target = case['expected_output']
            
            features.append([days, miles, receipts])
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def engineer_features(self, X):
        """Create engineered features from base inputs"""
        days = X[:, 0]
        miles = X[:, 1]
        receipts = X[:, 2]
        
        # Basic derived features
        efficiency = np.divide(miles, days, out=np.zeros_like(miles), where=days!=0)
        daily_spending = np.divide(receipts, days, out=np.zeros_like(receipts), where=days!=0)
        
        # Polynomial features
        days_squared = days ** 2
        miles_squared = miles ** 2
        receipts_squared = receipts ** 2
        
        # Interaction terms
        days_miles = days * miles
        days_receipts = days * receipts
        miles_receipts = miles * receipts
        efficiency_receipts = efficiency * receipts
        
        # Categorical features
        trip_short = (days <= 2).astype(float)
        trip_medium = ((days > 2) & (days <= 7)).astype(float)
        trip_long = (days > 7).astype(float)
        
        efficiency_low = (efficiency < 50).astype(float)
        efficiency_medium = ((efficiency >= 50) & (efficiency < 150)).astype(float)
        efficiency_high = (efficiency >= 150).astype(float)
        
        receipt_low = (receipts < 500).astype(float)
        receipt_medium = ((receipts >= 500) & (receipts < 1500)).astype(float)
        receipt_high = (receipts >= 1500).astype(float)
        
        # Log transforms
        log_miles = np.log1p(miles)
        log_receipts = np.log1p(receipts)
        
        # Special case indicators
        five_day_trip = (days == 5).astype(float)
        small_receipts = ((receipts > 0) & (receipts < 30) & (days > 1)).astype(float)
        high_efficiency = ((efficiency >= 180) & (efficiency <= 220)).astype(float)
        rounding_quirk = (((receipts * 100) % 100 == 49) | ((receipts * 100) % 100 == 99)).astype(float)
        
        # Combine all features
        engineered_features = np.column_stack([
            days, miles, receipts,
            efficiency, daily_spending,
            days_squared, miles_squared, receipts_squared,
            days_miles, days_receipts, miles_receipts, efficiency_receipts,
            trip_short, trip_medium, trip_long,
            efficiency_low, efficiency_medium, efficiency_high,
            receipt_low, receipt_medium, receipt_high,
            log_miles, log_receipts,
            five_day_trip, small_receipts, high_efficiency, rounding_quirk
        ])
        
        return engineered_features
    
    def train_models(self, X, y):
        """Train multiple ML models silently"""
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Initialize models with best parameters from previous testing
        self.models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # Train models
        for model in self.models.values():
            model.fit(X_scaled, y)
        
        # Create ensemble
        ensemble_models = [
            ('gb', self.models['gradient_boosting']),
            ('rf', self.models['random_forest'])
        ]
        
        self.models['ensemble'] = VotingRegressor(estimators=ensemble_models)
        self.models['ensemble'].fit(X_scaled, y)
        
        # Use ensemble as best model
        self.best_model = self.models['ensemble']
        self.is_trained = True
    
    def create_specialized_models(self, X, y):
        """Create specialized models for high-error cases"""
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        days = X[:, 0]
        miles = X[:, 1]
        receipts = X[:, 2]
        efficiency = miles / np.maximum(days, 1)
        
        # High receipt model
        high_receipt_mask = receipts > 1000
        if np.sum(high_receipt_mask) > 10:
            self.models['high_receipt'] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.models['high_receipt'].fit(X_scaled[high_receipt_mask], y[high_receipt_mask])
        
        # Low efficiency model
        low_efficiency_mask = efficiency < 30
        if np.sum(low_efficiency_mask) > 10:
            self.models['low_efficiency'] = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )
            self.models['low_efficiency'].fit(X_scaled[low_efficiency_mask], y[low_efficiency_mask])
        
        # Short high-mile model
        short_high_mile_mask = (days <= 2) & (miles > 300)
        if np.sum(short_high_mile_mask) > 10:
            self.models['short_high_mile'] = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )
            self.models['short_high_mile'].fit(X_scaled[short_high_mile_mask], y[short_high_mile_mask])
    
    def predict(self, days, miles, receipts):
        """Make prediction using the trained model"""
        X = np.array([[days, miles, receipts]])
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        efficiency = miles / max(days, 1)
        
        # Route to appropriate specialized model if applicable
        if 'high_receipt' in self.models and receipts > 1000:
            prediction = self.models['high_receipt'].predict(X_scaled)[0]
        elif 'low_efficiency' in self.models and efficiency < 30:
            prediction = self.models['low_efficiency'].predict(X_scaled)[0]
        elif 'short_high_mile' in self.models and days <= 2 and miles > 300:
            prediction = self.models['short_high_mile'].predict(X_scaled)[0]
        else:
            prediction = self.best_model.predict(X_scaled)[0]
        
        return round(max(0, prediction), 2)

# Global model instance (trained once)
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = SilentReimbursementMLModel()
    return _model_instance

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ml_reimbursement_silent.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        model = get_model()
        
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = model.predict(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)