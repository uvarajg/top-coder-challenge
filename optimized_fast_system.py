#!/usr/bin/env python3

import sys
import json
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class OptimizedFastSystem:
    """Ultra-fast optimized system with pre-trained models"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.is_trained = False
        self.model_file = 'fast_model.pkl'
        self.scaler_file = 'fast_scaler.pkl'
    
    def engineer_features(self, X):
        """Streamlined feature engineering for speed"""
        days = X[:, 0].astype(float)
        miles = X[:, 1].astype(float)
        receipts = X[:, 2].astype(float)
        
        # Core derived features
        efficiency = np.divide(miles, days, out=np.zeros_like(miles, dtype=float), where=days!=0)
        daily_spending = np.divide(receipts, days, out=np.zeros_like(receipts, dtype=float), where=days!=0)
        miles_per_dollar = np.divide(miles, np.maximum(receipts, 1))
        
        # Key business features (reduced set for speed)
        is_short_trip = (days <= 2).astype(float)
        is_medium_trip = ((days >= 3) & (days <= 7)).astype(float)
        is_long_trip = (days >= 8).astype(float)
        is_sweet_spot_length = ((days >= 4) & (days <= 6)).astype(float)
        
        is_optimal_efficiency = ((efficiency >= 180) & (efficiency <= 220)).astype(float)
        is_high_efficiency = (efficiency >= 200).astype(float)
        is_low_efficiency = (efficiency < 30).astype(float)
        
        is_five_day_trip = (days == 5).astype(float)
        is_kevin_sweet_spot = ((days == 5) & (efficiency >= 180) & (daily_spending < 100)).astype(float)
        is_vacation_penalty = ((days >= 8) & (daily_spending > 150)).astype(float)
        
        is_small_receipt_penalty = ((receipts > 0) & (receipts < 30) & (days > 1)).astype(float)
        is_medium_receipts = ((receipts >= 500) & (receipts <= 1500)).astype(float)
        has_rounding_quirk = (((receipts * 100) % 100 == 49) | ((receipts * 100) % 100 == 99)).astype(float)
        
        # Polynomial features (essential ones only)
        days_squared = days ** 2
        efficiency_squared = efficiency ** 2
        log_receipts = np.log1p(receipts)
        
        # Combine essential features (reduced from 35+ to 20)
        features = np.column_stack([
            days, miles, receipts,
            efficiency, daily_spending, miles_per_dollar,
            is_short_trip, is_medium_trip, is_long_trip, is_sweet_spot_length,
            is_optimal_efficiency, is_high_efficiency, is_low_efficiency,
            is_five_day_trip, is_kevin_sweet_spot, is_vacation_penalty,
            is_small_receipt_penalty, is_medium_receipts, has_rounding_quirk,
            days_squared, efficiency_squared, log_receipts
        ])
        
        return features
    
    def apply_business_constraints(self, prediction, days, miles, receipts):
        """Fast business rule constraints"""
        efficiency = miles / max(days, 1)
        daily_spending = receipts / max(days, 1)
        
        # Kevin's sweet spot bonus
        if days == 5 and efficiency >= 180 and daily_spending < 100:
            prediction *= 1.12
        
        # Five day bonus
        if days == 5:
            prediction += 25.0
        
        # Vacation penalty
        if days >= 8 and daily_spending > 150:
            prediction *= 0.88
        
        # Lisa's rounding quirk
        if receipts > 0:
            cents = int((receipts * 100) % 100)
            if cents == 49 or cents == 99:
                prediction += 15.0
        
        # Efficiency bonuses/penalties
        if 180 <= efficiency <= 220:
            prediction += 35.0
        elif efficiency >= 300:
            prediction += 20.0
        elif efficiency < 20 and days > 1:
            prediction -= 20.0
        
        # Small receipt penalty
        if 0 < receipts < 30 and days > 1:
            prediction -= 25.0
        
        # Reasonable bounds
        min_amount = days * 50.0
        max_amount = days * 400 + miles * 1.2 + receipts * 0.9
        prediction = max(min_amount, min(prediction, max_amount))
        
        return prediction
    
    def save_model(self):
        """Save trained model and scaler"""
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        return False
    
    def load_data(self, filepath='public_cases.json'):
        """Load training data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        X = []
        y = []
        for case in data:
            inp = case['input']
            X.append([inp['trip_duration_days'], inp['miles_traveled'], inp['total_receipts_amount']])
            y.append(case['expected_output'])
        
        return np.array(X), np.array(y)
    
    def train_once(self):
        """Train model once and save"""
        if self.load_model():
            return
            
        print("Training optimized model (one-time setup)...")
        
        X, y = self.load_data()
        X_features = self.engineer_features(X)
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Simplified ensemble for speed
        models = [
            ('gb', GradientBoostingRegressor(
                n_estimators=150,  # Reduced from 300
                max_depth=6,       # Reduced from 8
                learning_rate=0.1,  # Increased for faster training
                subsample=0.9,
                random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=100,   # Reduced from 200
                max_depth=12,       # Reduced from 15
                min_samples_split=3,
                random_state=42
            )),
            ('ridge', Ridge(alpha=0.5))
        ]
        
        # Simplified weights
        weights = [2.0, 1.0, 0.5]
        
        self.model = VotingRegressor(estimators=models, weights=weights)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        self.save_model()
        
        print("Model training complete and saved!")
    
    def predict(self, days, miles, receipts):
        """Ultra-fast prediction"""
        if not self.is_trained:
            if not self.load_model():
                self.train_once()
        
        # Prepare features
        X = np.array([[days, miles, receipts]])
        X_features = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_features)
        
        # Get base prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Apply business constraints
        final_prediction = self.apply_business_constraints(prediction, days, miles, receipts)
        
        return round(max(0, final_prediction), 2)

# Global instance
_optimized_system = None

def get_optimized_system():
    global _optimized_system
    if _optimized_system is None:
        _optimized_system = OptimizedFastSystem()
        _optimized_system.train_once()
    return _optimized_system

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 optimized_fast_system.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        system = get_optimized_system()
        
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = system.predict(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)