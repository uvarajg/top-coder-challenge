#!/usr/bin/env python3

import json
import numpy as np
import time
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

def main():
    print("🧾 FINAL SYSTEM EVALUATION")
    print("=" * 40)
    
    # Load test data
    X, y = load_test_data()
    print(f"Loaded {len(X)} test cases")
    
    # Test Fast Advanced system (our best performing system)
    print(f"\n🔍 Evaluating Fast Advanced Business Rule Engine...")
    
    start_time = time.time()
    try:
        fast_system = get_fast_system()
        
        predictions = []
        successful_runs = 0
        
        for i, (features, expected) in enumerate(zip(X, y)):
            try:
                prediction = fast_system.predict(features[0], features[1], features[2])
                predictions.append(prediction)
                successful_runs += 1
            except Exception as e:
                print(f"Error in case {i+1}: {e}")
                predictions.append(0)
        
        end_time = time.time()
        
        if successful_runs > 0:
            predictions = np.array(predictions[:successful_runs])
            y_actual = y[:successful_runs]
            
            # Calculate metrics
            exact_matches = np.sum(np.abs(predictions - y_actual) < 0.01)
            very_close_matches = np.sum(np.abs(predictions - y_actual) < 0.10)
            close_matches = np.sum(np.abs(predictions - y_actual) < 1.0)
            avg_error = np.mean(np.abs(predictions - y_actual))
            max_error = np.max(np.abs(predictions - y_actual))
            score = avg_error * 100 + (1000 - exact_matches) * 0.1
            
            print(f"\n✅ FINAL RESULTS:")
            print(f"  Total cases: {len(y)}")
            print(f"  Successful runs: {successful_runs}")
            print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/successful_runs*100:.1f}%)")
            print(f"  Very close matches (±$0.10): {very_close_matches} ({very_close_matches/successful_runs*100:.1f}%)")
            print(f"  Close matches (±$1.00): {close_matches} ({close_matches/successful_runs*100:.1f}%)")
            print(f"  Average error: ${avg_error:.2f}")
            print(f"  Maximum error: ${max_error:.2f}")
            print(f"  Score: {score:.2f}")
            print(f"  Execution time: {end_time - start_time:.1f}s")
            
            # Compare to original baseline
            baseline_exact = 1
            baseline_error = 183.13
            baseline_score = 18412.90
            
            exact_improvement = exact_matches / baseline_exact
            error_improvement = (baseline_error - avg_error) / baseline_error * 100
            score_improvement = (baseline_score - score) / baseline_score * 100
            
            print(f"\n🚀 IMPROVEMENT FROM ORIGINAL SYSTEM:")
            print(f"  Exact matches: {exact_improvement:.0f}x improvement ({baseline_exact} → {exact_matches})")
            print(f"  Average error: {error_improvement:.1f}% reduction (${baseline_error:.2f} → ${avg_error:.2f})")
            print(f"  Score: {score_improvement:.1f}% improvement ({baseline_score:.2f} → {score:.2f})")
            
            # Performance assessment
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            if exact_matches >= 20:
                print("🏆 EXCELLENT: Achieved significant exact matches!")
            elif exact_matches >= 10:
                print("🥇 GREAT: Strong performance on exact matches!")
            elif exact_matches >= 5:
                print("🥈 GOOD: Solid improvement in exact matches!")
            else:
                print("📈 Progress made, continue optimizing for exact matches")
                
            if avg_error < 50:
                print("✅ Low average error achieved")
            if score < 5000:
                print("✅ Excellent overall score")
                
            return {
                'exact_matches': exact_matches,
                'avg_error': avg_error,
                'score': score,
                'success': True
            }
        else:
            print("❌ No successful predictions")
            return {'success': False}
            
    except Exception as e:
        print(f"❌ System evaluation failed: {e}")
        return {'success': False}

if __name__ == "__main__":
    result = main()
    if result['success']:
        print(f"\n🎉 Final system successfully trained and evaluated!")
    else:
        print(f"\n❌ System evaluation incomplete")