#!/usr/bin/env python3

import json
import numpy as np
import time
from ultra_fast_system import ultra_fast_predict

def ultra_fast_evaluation():
    """Ultra-fast evaluation in under 1 second"""
    print("⚡ Ultra-Fast Evaluation")
    print("=" * 30)
    
    start_time = time.time()
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Run predictions
    predictions = []
    actuals = []
    
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        prediction = ultra_fast_predict(days, miles, receipts)
        predictions.append(prediction)
        actuals.append(expected)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    errors = np.abs(predictions - actuals)
    exact_matches = np.sum(errors < 0.01)
    close_matches = np.sum(errors < 1.0)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Calculate score (same as original eval.sh)
    score = avg_error * 100 + (len(actuals) - exact_matches) * 0.1
    
    end_time = time.time()
    
    print(f"Cases: {len(actuals)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    print(f"Avg error: ${avg_error:.2f}")
    print(f"Score: {score:.2f}")
    print(f"Time: {end_time - start_time:.3f}s")
    
    return score

if __name__ == "__main__":
    ultra_fast_evaluation()