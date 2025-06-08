#!/usr/bin/env python3

import json
import numpy as np
import time
import subprocess
import sys

def run_optimized_evaluation():
    """Run fast evaluation using optimized system"""
    
    print("Loading test cases...")
    start_time = time.time()
    
    # Load all test data at once
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"Running {len(data)} test cases...")
    
    # Prepare arrays for vectorized operations
    predictions = []
    actuals = []
    successful_runs = 0
    
    # Process in batches for better performance
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        for case in batch:
            inp = case['input']
            days = inp['trip_duration_days']
            miles = inp['miles_traveled'] 
            receipts = inp['total_receipts_amount']
            expected = case['expected_output']
            
            try:
                # Call optimized system directly
                result = subprocess.run([
                    'python3', 'optimized_fast_system.py', 
                    str(days), str(miles), str(receipts)
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    prediction = float(result.stdout.strip())
                    predictions.append(prediction)
                    actuals.append(expected)
                    successful_runs += 1
                else:
                    print(f"Error in case {i//batch_size * batch_size + len([c for c in batch if c == case])}: {result.stderr}")
                    
            except Exception as e:
                print(f"Exception in case {i//batch_size * batch_size + len([c for c in batch if c == case])}: {e}")
        
        # Progress update
        if i % 500 == 0:
            print(f"Progress: {min(i + batch_size, len(data))}/{len(data)} cases processed...")
    
    if successful_runs == 0:
        print("❌ No successful predictions!")
        return
    
    # Convert to numpy arrays for fast computation
    predictions = np.array(predictions)
    actuals = np.array(actuals[:successful_runs])
    
    # Calculate all metrics vectorized
    errors = np.abs(predictions - actuals)
    exact_matches = np.sum(errors < 0.01)
    very_close_matches = np.sum(errors < 0.10)
    close_matches = np.sum(errors < 1.0)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Calculate score
    score = avg_error * 100 + (len(data) - exact_matches) * 0.1
    
    end_time = time.time()
    
    # Display results
    print(f"\n✅ Evaluation Complete!")
    print(f"")
    print(f"📈 Results Summary:")
    print(f"  Total test cases: {len(data)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/successful_runs*100:.1f}%)")
    print(f"  Very close (±$0.10): {very_close_matches} ({very_close_matches/successful_runs*100:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/successful_runs*100:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Score: {score:.2f}")
    print(f"  Execution time: {end_time - start_time:.2f}s")
    
    # Performance assessment
    if exact_matches >= 20:
        print(f"\n🏆 EXCELLENT: {exact_matches} exact matches achieved!")
    elif exact_matches >= 10:
        print(f"\n🥇 GREAT: {exact_matches} exact matches!")
    elif exact_matches >= 5:
        print(f"\n🥈 GOOD: {exact_matches} exact matches!")
    elif exact_matches >= 1:
        print(f"\n🥉 PROGRESS: {exact_matches} exact matches!")
    else:
        print(f"\n📈 Continue optimizing for exact matches")
    
    if avg_error < 50:
        print("✅ Excellent average error achieved")
    if score < 6000:
        print("✅ Good overall score")
    
    return {
        'exact_matches': exact_matches,
        'avg_error': avg_error,
        'score': score,
        'execution_time': end_time - start_time
    }

if __name__ == "__main__":
    run_optimized_evaluation()