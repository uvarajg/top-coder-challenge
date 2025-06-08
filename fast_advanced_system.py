#!/usr/bin/env python3

import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class FastBusinessRuleEngine:
    """Fast implementation of the advanced business rule engine"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.is_trained = False
    
    def engineer_features(self, X):
        """Advanced feature engineering based on deep analysis"""
        days = X[:, 0]
        miles = X[:, 1] 
        receipts = X[:, 2]
        
        # Core derived features
        efficiency = np.divide(miles, days, out=np.zeros_like(miles), where=days!=0)
        daily_spending = np.divide(receipts, days, out=np.zeros_like(receipts), where=days!=0)
        miles_per_dollar = np.divide(miles, np.maximum(receipts, 1))
        
        # Tier 1: Core business features
        is_short_trip = (days <= 2).astype(float)
        is_medium_trip = ((days >= 3) & (days <= 7)).astype(float)
        is_long_trip = (days >= 8).astype(float)
        is_sweet_spot_length = ((days >= 4) & (days <= 6)).astype(float)
        
        is_low_efficiency = (efficiency < 30).astype(float)
        is_optimal_efficiency = ((efficiency >= 180) & (efficiency <= 220)).astype(float)
        is_high_efficiency = (efficiency >= 200).astype(float)
        
        is_small_receipt_penalty = ((receipts > 0) & (receipts < 30) & (days > 1)).astype(float)
        is_medium_receipts = ((receipts >= 500) & (receipts <= 1500)).astype(float)
        is_high_receipts = (receipts > 1500).astype(float)
        
        # Tier 2: Business rule features
        is_five_day_trip = (days == 5).astype(float)
        is_kevin_sweet_spot = ((days == 5) & (efficiency >= 180) & (daily_spending < 100)).astype(float)
        is_vacation_penalty = ((days >= 8) & (daily_spending > 150)).astype(float)
        
        exceeds_short_spending = ((days <= 3) & (daily_spending > 75)).astype(float)
        exceeds_medium_spending = ((days >= 4) & (days <= 6) & (daily_spending > 120)).astype(float)
        exceeds_long_spending = ((days >= 7) & (daily_spending > 90)).astype(float)
        
        has_rounding_quirk = (((receipts * 100) % 100 == 49) | ((receipts * 100) % 100 == 99)).astype(float)
        is_mileage_threshold = ((np.abs(miles - 100) < 5) | (np.abs(miles - 500) < 5) | (np.abs(miles - 1000) < 5)).astype(float)
        
        # Tier 3: Proxy features
        pseudo_seasonal = (np.array([hash(int(r * 100)) % 4 for r in receipts])).astype(float)
        employee_proxy = (np.array([hash((int(e * 10), int(ds))) % 10 for e, ds in zip(efficiency, daily_spending)])).astype(float)
        complexity_score = np.log1p(days * miles * receipts)
        
        # Advanced interactions
        efficiency_spending_interaction = efficiency * daily_spending / 1000  # Normalized
        high_value_short_trip = ((days <= 3) & (miles > 300)).astype(float)
        balanced_trip = ((days >= 4) & (days <= 6) & (efficiency >= 50) & (efficiency <= 150)).astype(float)
        
        # Polynomial features
        days_squared = days ** 2
        efficiency_squared = efficiency ** 2
        log_receipts = np.log1p(receipts)
        sqrt_efficiency = np.sqrt(np.maximum(efficiency, 0))
        
        # Combine features
        features = np.column_stack([
            # Base
            days, miles, receipts,
            # Derived
            efficiency, daily_spending, miles_per_dollar,
            # Tier 1
            is_short_trip, is_medium_trip, is_long_trip, is_sweet_spot_length,
            is_low_efficiency, is_optimal_efficiency, is_high_efficiency,
            is_small_receipt_penalty, is_medium_receipts, is_high_receipts,
            # Tier 2
            is_five_day_trip, is_kevin_sweet_spot, is_vacation_penalty,
            exceeds_short_spending, exceeds_medium_spending, exceeds_long_spending,
            has_rounding_quirk, is_mileage_threshold,
            # Tier 3
            pseudo_seasonal, employee_proxy, complexity_score,
            # Interactions
            efficiency_spending_interaction, high_value_short_trip, balanced_trip,
            # Polynomial
            days_squared, efficiency_squared, log_receipts, sqrt_efficiency
        ])
        
        return features
    
    def apply_business_constraints(self, prediction, days, miles, receipts):
        """Apply business rule constraints"""
        efficiency = miles / max(days, 1)
        daily_spending = receipts / max(days, 1)
        
        # Kevin's sweet spot bonus
        if days == 5 and efficiency >= 180 and daily_spending < 100:
            prediction *= 1.12
        
        # Vacation penalty
        if days >= 8 and daily_spending > 150:
            prediction *= 0.88
        
        # Lisa's rounding quirk
        if receipts > 0:
            cents = int((receipts * 100) % 100)
            if cents == 49 or cents == 99:
                prediction += 15.0
        
        # Five day bonus
        if days == 5:
            prediction += 25.0
        
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
    
    def train(self, X, y):
        """Train the fast advanced system"""
        print("Training fast advanced business rule engine...")
        
        # Engineer features
        X_features = self.engineer_features(X)
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Create optimized ensemble
        models = [
            ('gb1', GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.08, 
                subsample=0.9, random_state=42
            )),
            ('gb2', GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.15, 
                subsample=0.8, random_state=43
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=3,
                random_state=42
            )),
            ('ridge', Ridge(alpha=0.5))
        ]
        
        # Weight ensemble towards gradient boosting for exact matches
        weights = [2.0, 1.5, 1.0, 0.5]  # Favor GB models
        
        self.model = VotingRegressor(estimators=models, weights=weights)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Quick evaluation
        predictions = []
        for i, features in enumerate(X):
            pred = self.predict(features[0], features[1], features[2])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        exact_matches = np.sum(np.abs(predictions - y) < 0.01)
        close_matches = np.sum(np.abs(predictions - y) < 1.0)
        avg_error = np.mean(np.abs(predictions - y))
        
        print(f"Training complete!")
        print(f"Exact matches: {exact_matches}/{len(y)} ({exact_matches/len(y)*100:.1f}%)")
        print(f"Close matches: {close_matches}/{len(y)} ({close_matches/len(y)*100:.1f}%)")
        print(f"Average error: ${avg_error:.2f}")
        
        return {
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'avg_error': avg_error
        }
    
    def predict(self, days, miles, receipts):
        """Make prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
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
_fast_system = None

def get_fast_system():
    global _fast_system
    if _fast_system is None:
        _fast_system = FastBusinessRuleEngine()
        X, y = _fast_system.load_data()
        _fast_system.train(X, y)
    return _fast_system

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 fast_advanced_system.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        system = get_fast_system()
        
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = system.predict(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)