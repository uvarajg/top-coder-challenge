#!/usr/bin/env python3

import json
import numpy as np
import time
from optimized_fast_system import get_optimized_system

def fast_evaluation():
    """Ultra-fast evaluation using pre-trained model"""
    print("⚡ Fast Evaluation - Optimized System")
    print("=" * 40)
    
    start_time = time.time()
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Get pre-trained system
    system = get_optimized_system()
    
    # Run predictions
    predictions = []
    actuals = []
    
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        prediction = system.predict(days, miles, receipts)
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
    
    print(f"Total cases: {len(actuals)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    print(f"Average error: {avg_error:.2f}")
    print(f"Max error: {max_error:.2f}")
    print(f"Score: {score:.2f}")
    print(f"Execution time: {end_time - start_time:.2f}s")
    
    return score

if __name__ == "__main__":
    fast_evaluation()