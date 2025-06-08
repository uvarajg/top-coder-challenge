#!/usr/bin/env python3

import json
import numpy as np
import time
from ml_reimbursement_silent import get_ml_system
from fast_advanced_system import get_fast_system

def load_test_data():
    """Load test data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    for case in data:
        inp = case['input']
        X.append([inp['trip_duration_days'], inp['miles_traveled'], inp['total_receipts_amount']])
        y.append(case['expected_output'])
    
    return np.array(X), np.array(y)

def evaluate_system(system_name, predict_func, X, y):
    """Evaluate a system's performance"""
    print(f"\n🔍 Evaluating {system_name}...")
    
    start_time = time.time()
    predictions = []
    successful_runs = 0
    
    for i, (features, expected) in enumerate(zip(X, y)):
        try:
            prediction = predict_func(features[0], features[1], features[2])
            predictions.append(prediction)
            successful_runs += 1
        except Exception as e:
            print(f"Error in case {i+1}: {e}")
            predictions.append(0)
    
    end_time = time.time()
    
    if successful_runs == 0:
        return None
    
    predictions = np.array(predictions[:successful_runs])
    y_actual = y[:successful_runs]
    
    # Calculate metrics
    exact_matches = np.sum(np.abs(predictions - y_actual) < 0.01)
    very_close_matches = np.sum(np.abs(predictions - y_actual) < 0.10)
    close_matches = np.sum(np.abs(predictions - y_actual) < 1.0)
    avg_error = np.mean(np.abs(predictions - y_actual))
    max_error = np.max(np.abs(predictions - y_actual))
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"✅ {system_name} Results:")
    print(f"  Total cases: {len(y)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/successful_runs*100:.1f}%)")
    print(f"  Very close matches (±$0.10): {very_close_matches} ({very_close_matches/successful_runs*100:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/successful_runs*100:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Score: {score:.2f}")
    print(f"  Execution time: {end_time - start_time:.1f}s")
    
    return {
        'system_name': system_name,
        'successful_runs': successful_runs,
        'exact_matches': exact_matches,
        'very_close_matches': very_close_matches,
        'close_matches': close_matches,
        'avg_error': avg_error,
        'max_error': max_error,
        'score': score,
        'execution_time': end_time - start_time
    }

def main():
    print("🧾 COMPREHENSIVE SYSTEM EVALUATION")
    print("=" * 50)
    
    # Load test data
    X, y = load_test_data()
    print(f"Loaded {len(X)} test cases")
    
    results = []
    
    # Test ML system
    try:
        ml_system = get_ml_system()
        result = evaluate_system("ML Enhanced System", ml_system.predict, X, y)
        if result:
            results.append(result)
    except Exception as e:
        print(f"❌ ML System failed: {e}")
    
    # Test Fast Advanced system
    try:
        fast_system = get_fast_system()
        result = evaluate_system("Fast Advanced System", fast_system.predict, X, y)
        if result:
            results.append(result)
    except Exception as e:
        print(f"❌ Fast Advanced System failed: {e}")
    
    # Summary comparison
    if results:
        print("\n" + "=" * 70)
        print("📊 FINAL COMPARISON")
        print("=" * 70)
        
        print(f"{'System':<25} {'Exact':<8} {'Close':<8} {'Avg Error':<12} {'Score':<10} {'Time':<8}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['system_name']:<25} {result['exact_matches']:<8} {result['close_matches']:<8} "
                  f"${result['avg_error']:<11.2f} {result['score']:<10.2f} {result['execution_time']:<8.1f}s")
        
        # Find best system
        best_exact = max(results, key=lambda x: x['exact_matches'])
        best_error = min(results, key=lambda x: x['avg_error'])
        best_score = min(results, key=lambda x: x['score'])
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"  Most exact matches: {best_exact['system_name']} ({best_exact['exact_matches']})")
        print(f"  Lowest average error: {best_error['system_name']} (${best_error['avg_error']:.2f})")
        print(f"  Best score: {best_score['system_name']} ({best_score['score']:.2f})")
        
        # Calculate improvements over baseline
        baseline_exact = 1  # Original system had 1 exact match
        baseline_error = 183.13  # Original system error
        baseline_score = 18412.90  # Original system score
        
        best_system = min(results, key=lambda x: x['score'])
        exact_improvement = best_system['exact_matches'] / baseline_exact
        error_improvement = (baseline_error - best_system['avg_error']) / baseline_error * 100
        score_improvement = (baseline_score - best_system['score']) / baseline_score * 100
        
        print(f"\n🚀 OVERALL IMPROVEMENT FROM ORIGINAL:")
        print(f"  Exact matches: {exact_improvement:.0f}x improvement")
        print(f"  Average error: {error_improvement:.1f}% reduction")
        print(f"  Score: {score_improvement:.1f}% improvement")

if __name__ == "__main__":
    main()