#!/usr/bin/env python3

import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ReimbursementMLModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_data(self, filepath='public_cases.json'):
        """Load and prepare training data from public cases"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract features and targets
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
        
        # Categorical features (one-hot encoded)
        trip_short = (days <= 2).astype(float)
        trip_medium = ((days > 2) & (days <= 7)).astype(float)
        trip_long = (days > 7).astype(float)
        
        efficiency_low = (efficiency < 50).astype(float)
        efficiency_medium = ((efficiency >= 50) & (efficiency < 150)).astype(float)
        efficiency_high = (efficiency >= 150).astype(float)
        
        receipt_low = (receipts < 500).astype(float)
        receipt_medium = ((receipts >= 500) & (receipts < 1500)).astype(float)
        receipt_high = (receipts >= 1500).astype(float)
        
        # Log transforms for skewed data
        log_miles = np.log1p(miles)
        log_receipts = np.log1p(receipts)
        
        # Special case indicators
        five_day_trip = (days == 5).astype(float)
        small_receipts = ((receipts > 0) & (receipts < 30) & (days > 1)).astype(float)
        high_efficiency = ((efficiency >= 180) & (efficiency <= 220)).astype(float)
        rounding_quirk = (((receipts * 100) % 100 == 49) | ((receipts * 100) % 100 == 99)).astype(float)
        
        # Combine all features
        engineered_features = np.column_stack([
            # Base features
            days, miles, receipts,
            
            # Derived features
            efficiency, daily_spending,
            
            # Polynomial features
            days_squared, miles_squared, receipts_squared,
            
            # Interaction terms
            days_miles, days_receipts, miles_receipts, efficiency_receipts,
            
            # Categorical features
            trip_short, trip_medium, trip_long,
            efficiency_low, efficiency_medium, efficiency_high,
            receipt_low, receipt_medium, receipt_high,
            
            # Log transforms
            log_miles, log_receipts,
            
            # Special indicators
            five_day_trip, small_receipts, high_efficiency, rounding_quirk
        ])
        
        return engineered_features
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("Training machine learning models...")
        
        # Engineer features
        X_engineered = self.engineer_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Initialize models
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
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
        }
        
        # Train models and evaluate
        cv_scores = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            
            # Cross-validation score
            cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
            cv_scores[name] = -cv_score.mean()
            print(f"{name} CV MAE: ${cv_scores[name]:.2f}")
        
        # Create ensemble
        ensemble_models = [
            ('gb', self.models['gradient_boosting']),
            ('rf', self.models['random_forest']),
            ('nn', self.models['neural_network'])
        ]
        
        self.models['ensemble'] = VotingRegressor(
            estimators=ensemble_models,
            weights=[1, 1, 0.5]  # Weight neural network less due to potential overfitting
        )
        
        print("Training ensemble model...")
        self.models['ensemble'].fit(X_scaled, y)
        
        # Evaluate ensemble
        ensemble_score = cross_val_score(self.models['ensemble'], X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        cv_scores['ensemble'] = -ensemble_score.mean()
        print(f"ensemble CV MAE: ${cv_scores['ensemble']:.2f}")
        
        # Select best model
        best_model_name = min(cv_scores, key=cv_scores.get)
        self.best_model = self.models[best_model_name]
        print(f"Best model: {best_model_name} with MAE: ${cv_scores[best_model_name]:.2f}")
        
        self.is_trained = True
        return cv_scores
    
    def create_specialized_models(self, X, y):
        """Create specialized models for high-error cases"""
        print("Creating specialized models for high-error cases...")
        
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        days = X[:, 0]
        miles = X[:, 1]
        receipts = X[:, 2]
        efficiency = miles / np.maximum(days, 1)
        
        # High receipt model (for cases like your high-error examples)
        high_receipt_mask = receipts > 1000
        if np.sum(high_receipt_mask) > 10:  # Need minimum samples
            print(f"Training high-receipt model on {np.sum(high_receipt_mask)} cases...")
            self.models['high_receipt'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['high_receipt'].fit(X_scaled[high_receipt_mask], y[high_receipt_mask])
        
        # Low efficiency model
        low_efficiency_mask = efficiency < 30
        if np.sum(low_efficiency_mask) > 10:
            print(f"Training low-efficiency model on {np.sum(low_efficiency_mask)} cases...")
            self.models['low_efficiency'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.models['low_efficiency'].fit(X_scaled[low_efficiency_mask], y[low_efficiency_mask])
        
        # Short high-mile model
        short_high_mile_mask = (days <= 2) & (miles > 300)
        if np.sum(short_high_mile_mask) > 10:
            print(f"Training short-high-mile model on {np.sum(short_high_mile_mask)} cases...")
            self.models['short_high_mile'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.models['short_high_mile'].fit(X_scaled[short_high_mile_mask], y[short_high_mile_mask])
    
    def predict(self, days, miles, receipts):
        """Make prediction using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
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
            # Use best general model
            prediction = self.best_model.predict(X_scaled)[0]
        
        # Apply business constraints
        prediction = max(0, prediction)
        
        return round(prediction, 2)
    
    def evaluate_on_public_cases(self):
        """Evaluate the model on public cases to estimate performance"""
        X, y = self.load_data()
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        predictions = []
        for i in range(len(X)):
            days, miles, receipts = X[i]
            pred = self.predict(days, miles, receipts)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        # Calculate exact and close matches
        errors = np.abs(predictions - y)
        exact_matches = np.sum(errors < 0.01)
        close_matches = np.sum(errors < 1.0)
        max_error = np.max(errors)
        
        print(f"\n=== ML Model Evaluation ===")
        print(f"Total test cases: {len(y)}")
        print(f"Exact matches (±$0.01): {exact_matches} ({exact_matches/len(y)*100:.1f}%)")
        print(f"Close matches (±$1.00): {close_matches} ({close_matches/len(y)*100:.1f}%)")
        print(f"Average error: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"Maximum error: ${max_error:.2f}")
        
        # Calculate score (same as eval.sh)
        score = mae * 100 + (len(y) - exact_matches) * 0.1
        print(f"Score: {score:.2f} (lower is better)")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'max_error': max_error,
            'score': score
        }

def train_ml_model():
    """Train the ML model and save it"""
    model = ReimbursementMLModel()
    
    # Load data
    X, y = model.load_data()
    print(f"Loaded {len(X)} training cases")
    
    # Train models
    cv_scores = model.train_models(X, y)
    
    # Create specialized models
    model.create_specialized_models(X, y)
    
    # Evaluate performance
    results = model.evaluate_on_public_cases()
    
    return model, results

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ml_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    # For prediction mode, we need to load a pre-trained model
    # For now, train on the fly (in production, we'd save/load the model)
    try:
        model = ReimbursementMLModel()
        X, y = model.load_data()
        model.train_models(X, y)
        model.create_specialized_models(X, y)
        
        trip_duration = sys.argv[1]
        miles = sys.argv[2]
        receipts = sys.argv[3]
        
        result = model.predict(float(trip_duration), float(miles), float(receipts))
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)