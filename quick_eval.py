#!/usr/bin/env python3

import json
import numpy as np
import time
from ml_reimbursement_optimized import get_optimized_model

def quick_evaluation():
    """Quick evaluation on first 100 cases to check performance"""
    print("⚡ Quick Performance Check (100 cases)")
    print("=" * 40)
    
    start_time = time.time()
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Use first 100 cases for quick check
    test_cases = data[:100]
    
    # Get pre-trained system
    model = get_optimized_model()
    
    # Run predictions
    predictions = []
    actuals = []
    
    for case in test_cases:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        prediction = model.predict(days, miles, receipts)
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
    
    end_time = time.time()
    
    print(f"Cases tested: {len(actuals)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Max error: ${max_error:.2f}")
    print(f"Execution time: {end_time - start_time:.2f}s")
    
    # Extrapolate to full 1000 cases
    projected_exact = exact_matches * 10
    projected_avg_error = avg_error
    projected_score = projected_avg_error * 100 + (1000 - projected_exact) * 0.1
    
    print(f"\n📈 Projected Full Performance:")
    print(f"  Projected exact matches: ~{projected_exact}")
    print(f"  Projected avg error: ~${projected_avg_error:.2f}")
    print(f"  Projected score: ~{projected_score:.2f}")
    
    if projected_avg_error < 35:
        print("✅ RESTORED: Performance looks good!")
    else:
        print("⚠️  Performance still not optimal")
    
    return {
        'avg_error': avg_error,
        'exact_matches': exact_matches,
        'execution_time': end_time - start_time
    }

if __name__ == "__main__":
    quick_evaluation()