#!/usr/bin/env python3

import sys
import json
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedReimbursementMLModel:
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.is_trained = False
        self.cache_dir = 'ml_cache_enhanced'
        self.model_file = os.path.join(self.cache_dir, 'enhanced_models.pkl')
        self.scaler_file = os.path.join(self.cache_dir, 'enhanced_scaler.pkl')
        
        # Try to load cached models first, train if not available
        if not self.load_cached_models():
            self._train_once()
            self.save_cached_models()
        
    def load_cached_models(self):
        """Load pre-trained models from cache"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.models = pickle.load(f)
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.best_model = self.models.get('ensemble')
                self.is_trained = True
                return True
            except:
                return False
        return False
    
    def save_cached_models(self):
        """Save trained models to cache"""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.models, f)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
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
    
    def engineer_enhanced_features(self, X):
        """Enhanced feature engineering targeting identified weak points"""
        days = X[:, 0].astype(float)
        miles = X[:, 1].astype(float)
        receipts = X[:, 2].astype(float)
        
        # Core derived features
        efficiency = np.divide(miles, days, out=np.zeros_like(miles, dtype=float), where=days!=0)
        daily_spending = np.divide(receipts, days, out=np.zeros_like(receipts, dtype=float), where=days!=0)
        miles_per_dollar = np.divide(miles, np.maximum(receipts, 1))
        
        # === ENHANCED FEATURES FOR PROBLEM CASES ===
        
        # 1. EXTREME EFFICIENCY HANDLING (Cases with very low/high efficiency)
        ultra_low_efficiency = (efficiency < 10).astype(float)
        very_low_efficiency = ((efficiency >= 10) & (efficiency < 30)).astype(float)
        low_efficiency = ((efficiency >= 30) & (efficiency < 50)).astype(float)
        normal_efficiency = ((efficiency >= 50) & (efficiency < 150)).astype(float)
        high_efficiency = ((efficiency >= 150) & (efficiency < 300)).astype(float)
        ultra_high_efficiency = (efficiency >= 300).astype(float)
        
        # 2. SINGLE DAY TRIP SPECIAL HANDLING
        single_day_trip = (days == 1).astype(float)
        single_day_high_miles = ((days == 1) & (miles > 400)).astype(float)
        single_day_high_spending = ((days == 1) & (receipts > 500)).astype(float)
        single_day_extreme = ((days == 1) & (miles > 800) & (receipts > 1000)).astype(float)
        
        # 3. LONG TRIP NUANCED HANDLING
        very_long_trip = (days >= 10).astype(float)
        ultra_long_trip = (days >= 14).astype(float)
        long_trip_low_efficiency = ((days >= 8) & (efficiency < 50)).astype(float)
        long_trip_low_spending = ((days >= 8) & (daily_spending < 100)).astype(float)
        
        # 4. RECEIPT ANOMALY DETECTION
        receipt_per_mile = np.divide(receipts, np.maximum(miles, 1))
        high_receipt_per_mile = (receipt_per_mile > 2.0).astype(float)
        very_high_receipts_short = ((days <= 3) & (receipts > 1500)).astype(float)
        disproportionate_receipts = ((receipts > daily_spending * days * 1.5)).astype(float)
        
        # 5. BUSINESS RULE PATTERNS (Enhanced from interviews)
        kevin_optimal_range = ((efficiency >= 180) & (efficiency <= 220)).astype(float)
        five_day_optimal = ((days == 5) & (efficiency >= 180) & (daily_spending < 100)).astype(float)
        lisa_rounding_49 = ((receipts * 100) % 100 == 49).astype(float)
        lisa_rounding_99 = ((receipts * 100) % 100 == 99).astype(float)
        small_receipt_penalty = ((receipts > 0) & (receipts < 30) & (days > 1)).astype(float)
        
        # 6. INTERACTION FEATURES FOR COMPLEX PATTERNS
        efficiency_spending_ratio = np.divide(efficiency, np.maximum(daily_spending, 1))
        days_efficiency_interaction = days * efficiency
        receipts_efficiency_interaction = receipts * efficiency / 1000  # Normalized
        complex_trip_indicator = ((days > 7) & (miles > 500) & (receipts > 800)).astype(float)
        
        # 7. POLYNOMIAL AND TRANSFORMATIONS (Enhanced)
        days_squared = days ** 2
        days_cubed = days ** 3  # For very long trips
        miles_squared = miles ** 2
        receipts_squared = receipts ** 2
        efficiency_squared = efficiency ** 2
        
        log_days = np.log1p(days)
        log_miles = np.log1p(miles)
        log_receipts = np.log1p(receipts)
        log_efficiency = np.log1p(efficiency)
        
        sqrt_days = np.sqrt(days)
        sqrt_miles = np.sqrt(miles)
        sqrt_receipts = np.sqrt(receipts)
        sqrt_efficiency = np.sqrt(np.maximum(efficiency, 0))
        
        # 8. OUTLIER DETECTION FEATURES
        efficiency_zscore = np.abs((efficiency - np.mean(efficiency)) / np.maximum(np.std(efficiency), 1))
        spending_zscore = np.abs((daily_spending - np.mean(daily_spending)) / np.maximum(np.std(daily_spending), 1))
        is_efficiency_outlier = (efficiency_zscore > 2).astype(float)
        is_spending_outlier = (spending_zscore > 2).astype(float)
        
        # 9. CONTEXTUAL FEATURES (Simulated)
        trip_complexity = np.log1p(days * miles * receipts / 10000)
        normalized_efficiency = efficiency / np.maximum(np.mean(efficiency), 1)
        normalized_spending = daily_spending / np.maximum(np.mean(daily_spending), 1)
        
        # Combine all enhanced features (45+ features)
        enhanced_features = np.column_stack([
            # Base features
            days, miles, receipts,
            
            # Core derived
            efficiency, daily_spending, miles_per_dollar,
            
            # Enhanced efficiency categories
            ultra_low_efficiency, very_low_efficiency, low_efficiency,
            normal_efficiency, high_efficiency, ultra_high_efficiency,
            
            # Single day specialization
            single_day_trip, single_day_high_miles, single_day_high_spending, single_day_extreme,
            
            # Long trip specialization
            very_long_trip, ultra_long_trip, long_trip_low_efficiency, long_trip_low_spending,
            
            # Receipt anomalies
            receipt_per_mile, high_receipt_per_mile, very_high_receipts_short, disproportionate_receipts,
            
            # Business rules
            kevin_optimal_range, five_day_optimal, lisa_rounding_49, lisa_rounding_99, small_receipt_penalty,
            
            # Interactions
            efficiency_spending_ratio, days_efficiency_interaction, receipts_efficiency_interaction, complex_trip_indicator,
            
            # Polynomial
            days_squared, days_cubed, miles_squared, receipts_squared, efficiency_squared,
            
            # Transformations
            log_days, log_miles, log_receipts, log_efficiency,
            sqrt_days, sqrt_miles, sqrt_receipts, sqrt_efficiency,
            
            # Outlier detection
            is_efficiency_outlier, is_spending_outlier,
            
            # Contextual
            trip_complexity, normalized_efficiency, normalized_spending
        ])
        
        return enhanced_features
    
    def train_models(self, X, y):
        """Train enhanced ensemble with more diverse models"""
        X_engineered = self.engineer_enhanced_features(X)
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Enhanced model ensemble targeting different aspects
        self.models = {
            # Gradient boosting variants (best for complex patterns)
            'gradient_boosting_main': GradientBoostingRegressor(
                n_estimators=400,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                random_state=42,
                loss='absolute_error'  # Better for exact matches
            ),
            'gradient_boosting_robust': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                random_state=43,
                loss='huber'  # Robust to outliers
            ),
            
            # Random forest variants
            'random_forest_deep': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=2,
                random_state=42
            ),
            
            # Neural network for complex patterns
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            
            # Linear models for regularization
            'ridge': Ridge(alpha=0.5),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        # Train all models
        for model in self.models.values():
            model.fit(X_scaled, y)
        
        # Create weighted ensemble (favor GB and RF)
        ensemble_models = [
            ('gb_main', self.models['gradient_boosting_main']),
            ('gb_robust', self.models['gradient_boosting_robust']),
            ('rf_deep', self.models['random_forest_deep']),
            ('extra_trees', self.models['extra_trees']),
            ('nn', self.models['neural_network']),
            ('ridge', self.models['ridge'])
        ]
        
        # Weights favoring models good at exact matches
        weights = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        
        self.models['ensemble'] = VotingRegressor(estimators=ensemble_models, weights=weights)
        self.models['ensemble'].fit(X_scaled, y)
        
        self.best_model = self.models['ensemble']
        self.is_trained = True
    
    def create_specialized_models(self, X, y):
        """Create specialized models for identified problem areas"""
        X_engineered = self.engineer_enhanced_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        days = X[:, 0]
        miles = X[:, 1]
        receipts = X[:, 2]
        efficiency = miles / np.maximum(days, 1)
        daily_spending = receipts / np.maximum(days, 1)
        
        # 1. Ultra-low efficiency specialist
        ultra_low_eff_mask = efficiency < 10
        if np.sum(ultra_low_eff_mask) > 5:
            self.models['ultra_low_efficiency'] = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
            self.models['ultra_low_efficiency'].fit(X_scaled[ultra_low_eff_mask], y[ultra_low_eff_mask])
        
        # 2. Single day extreme cases specialist
        single_extreme_mask = (days == 1) & ((miles > 400) | (receipts > 500))
        if np.sum(single_extreme_mask) > 5:
            self.models['single_day_extreme'] = RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42
            )
            self.models['single_day_extreme'].fit(X_scaled[single_extreme_mask], y[single_extreme_mask])
        
        # 3. Very long trip specialist
        very_long_mask = days >= 10
        if np.sum(very_long_mask) > 10:
            self.models['very_long_trips'] = GradientBoostingRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42
            )
            self.models['very_long_trips'].fit(X_scaled[very_long_mask], y[very_long_mask])
        
        # 4. High receipt anomaly specialist
        high_receipt_mask = (receipts > 1500) | ((receipts / np.maximum(daily_spending * days, 1)) > 1.5)
        if np.sum(high_receipt_mask) > 10:
            self.models['high_receipt_anomaly'] = RandomForestRegressor(
                n_estimators=150, max_depth=10, random_state=42
            )
            self.models['high_receipt_anomaly'].fit(X_scaled[high_receipt_mask], y[high_receipt_mask])
    
    def apply_enhanced_business_rules(self, prediction, days, miles, receipts):
        """Enhanced business rule postprocessing"""
        efficiency = miles / max(days, 1)
        daily_spending = receipts / max(days, 1)
        
        # Kevin's sweet spot (enhanced)
        if days == 5 and efficiency >= 180 and daily_spending < 100:
            prediction *= 1.15
        
        # Five day bonus (Lisa confirmed)
        if days == 5:
            prediction += 30.0
        
        # Enhanced vacation penalty
        if days >= 8 and daily_spending > 150:
            prediction *= 0.85
        elif days >= 10 and daily_spending > 120:
            prediction *= 0.88
        elif days >= 14:
            prediction *= 0.82  # Extra penalty for ultra-long trips
        
        # Lisa's rounding quirks (enhanced detection)
        if receipts > 0:
            cents = int((receipts * 100) % 100)
            if cents == 49:
                prediction += 18.0
            elif cents == 99:
                prediction += 15.0
        
        # Enhanced efficiency bonuses/penalties
        if efficiency >= 300:  # Ultra-high efficiency
            prediction += 50.0
        elif 180 <= efficiency <= 220:  # Kevin's sweet spot
            prediction += 40.0
        elif efficiency >= 200:
            prediction += 25.0
        elif efficiency < 10 and days > 1:  # Ultra-low efficiency penalty
            prediction -= 40.0
        elif efficiency < 20 and days > 1:
            prediction -= 25.0
        
        # Enhanced small receipt penalty
        if 0 < receipts < 30 and days > 1:
            prediction -= 30.0
        elif 0 < receipts < 50 and days > 2:
            prediction -= 15.0
        
        # Single day adjustments
        if days == 1:
            if miles > 800:
                prediction += 60.0  # Extreme single day bonus
            elif miles > 400:
                prediction += 40.0
            else:
                prediction += 25.0
        
        # Trip length adjustments (enhanced)
        elif days <= 2:
            prediction += 35.0
        elif days >= 12:
            penalty = (days - 11) * 20  # Increased penalty
            prediction -= penalty
        
        # Ensure reasonable bounds (enhanced)
        min_reasonable = days * 35.0
        max_reasonable = days * 500 + miles * 2.0 + receipts * 1.2
        prediction = max(min_reasonable, min(prediction, max_reasonable))
        
        return prediction
    
    def predict(self, days, miles, receipts):
        """Enhanced prediction with specialized routing"""
        X = np.array([[days, miles, receipts]])
        X_engineered = self.engineer_enhanced_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        
        efficiency = miles / max(days, 1)
        daily_spending = receipts / max(days, 1)
        
        # Enhanced routing logic
        if 'ultra_low_efficiency' in self.models and efficiency < 10:
            prediction = self.models['ultra_low_efficiency'].predict(X_scaled)[0]
        elif 'single_day_extreme' in self.models and days == 1 and (miles > 400 or receipts > 500):
            prediction = self.models['single_day_extreme'].predict(X_scaled)[0]
        elif 'very_long_trips' in self.models and days >= 10:
            prediction = self.models['very_long_trips'].predict(X_scaled)[0]
        elif 'high_receipt_anomaly' in self.models and (receipts > 1500 or (receipts / max(daily_spending * days, 1)) > 1.5):
            prediction = self.models['high_receipt_anomaly'].predict(X_scaled)[0]
        else:
            prediction = self.best_model.predict(X_scaled)[0]
        
        # Apply enhanced business rules
        final_prediction = self.apply_enhanced_business_rules(prediction, days, miles, receipts)
        
        return round(max(0, final_prediction), 2)

# Global model instance
_enhanced_model_instance = None

def get_enhanced_model():
    global _enhanced_model_instance
    if _enhanced_model_instance is None:
        _enhanced_model_instance = EnhancedReimbursementMLModel()
    return _enhanced_model_instance

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ml_reimbursement_enhanced.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        model = get_enhanced_model()
        
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = model.predict(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)