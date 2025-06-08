#!/usr/bin/env python3

import json
import numpy as np
from ml_reimbursement import ReimbursementMLModel

def evaluate_ml_performance():
    """Evaluate ML model performance in the same format as eval.sh"""
    
    # Load the model and train it once
    print("🧾 Black Box Challenge - ML Model Evaluation")
    print("=======================================================")
    print()
    print("📊 Training ML model on 1,000 historical cases...")
    
    model = ReimbursementMLModel()
    X, y = model.load_data()
    
    # Train models
    cv_scores = model.train_models(X, y)
    model.create_specialized_models(X, y)
    
    print()
    print("📊 Running evaluation against 1,000 test cases...")
    print()
    
    # Now evaluate on all cases
    predictions = []
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    max_error = 0.0
    max_error_case = ""
    errors_list = []
    
    for i, (features, expected) in enumerate(zip(X, y)):
        days, miles, receipts = features
        
        try:
            prediction = model.predict(days, miles, receipts)
            error = abs(prediction - expected)
            
            predictions.append(prediction)
            successful_runs += 1
            
            # Check for exact match (within $0.01)
            if error < 0.01:
                exact_matches += 1
            
            # Check for close match (within $1.00)
            if error < 1.0:
                close_matches += 1
            
            total_error += error
            errors_list.append((i+1, expected, prediction, error, days, miles, receipts))
            
            # Track maximum error
            if error > max_error:
                max_error = error
                max_error_case = f"Case {i+1}: {days} days, {miles} miles, ${receipts} receipts"
                
        except Exception as e:
            print(f"Error in case {i+1}: {e}")
    
    # Calculate results
    if successful_runs > 0:
        avg_error = total_error / successful_runs
        exact_pct = exact_matches * 100 / successful_runs
        close_pct = close_matches * 100 / successful_runs
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        print("✅ Evaluation Complete!")
        print("")
        print("📈 Results Summary:")
        print(f"  Total test cases: 1000")
        print(f"  Successful runs: {successful_runs}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Maximum error: ${max_error:.2f}")
        print("")
        print(f"🎯 Your Score: {score:.2f} (lower is better)")
        print("")
        
        # Provide feedback based on exact matches
        if exact_matches == 1000:
            print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
        elif exact_matches > 950:
            print("🥇 Excellent! You are very close to the perfect solution.")
        elif exact_matches > 800:
            print("🥈 Great work! You have captured most of the system behavior.")
        elif exact_matches > 500:
            print("🥉 Good progress! You understand some key patterns.")
        else:
            print("📚 Keep analyzing the patterns in the interviews and test cases.")
        
        print("")
        print("💡 Tips for improvement:")
        if exact_matches < 1000:
            print("  Check these high-error cases:")
            
            # Sort by error and show top 5
            errors_list.sort(key=lambda x: x[3], reverse=True)
            for i in range(min(5, len(errors_list))):
                case_num, expected, actual, error, days, miles, receipts = errors_list[i]
                print(f"    Case {case_num}: {days} days, {miles} miles, ${receipts} receipts")
                print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
        
        return {
            'successful_runs': successful_runs,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'avg_error': avg_error,
            'max_error': max_error,
            'score': score,
            'predictions': predictions
        }
    else:
        print("❌ No successful test cases!")
        return None

def compare_with_original():
    """Compare ML performance with original rule-based approach"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    # Original performance (from your eval.sh results)
    original = {
        'exact_matches': 1,
        'close_matches': 7,
        'avg_error': 183.13,
        'score': 18412.90
    }
    
    # ML performance
    ml_results = evaluate_ml_performance()
    
    if ml_results:
        print(f"\n📈 IMPROVEMENT METRICS:")
        print(f"Exact matches: {original['exact_matches']} → {ml_results['exact_matches']} ({ml_results['exact_matches'] - original['exact_matches']:+d})")
        print(f"Close matches: {original['close_matches']} → {ml_results['close_matches']} ({ml_results['close_matches'] - original['close_matches']:+d})")
        print(f"Average error: ${original['avg_error']:.2f} → ${ml_results['avg_error']:.2f} ({ml_results['avg_error'] - original['avg_error']:+.2f})")
        print(f"Score: {original['score']:.2f} → {ml_results['score']:.2f} ({ml_results['score'] - original['score']:+.2f})")
        
        # Calculate improvement percentages
        error_improvement = (original['avg_error'] - ml_results['avg_error']) / original['avg_error'] * 100
        score_improvement = (original['score'] - ml_results['score']) / original['score'] * 100
        
        print(f"\n🎯 IMPROVEMENT SUMMARY:")
        print(f"Average error reduced by {error_improvement:.1f}%")
        print(f"Score improved by {score_improvement:.1f}%")
        print(f"Exact matches increased by {ml_results['exact_matches'] - original['exact_matches']}x")
        print(f"Close matches increased by {(ml_results['close_matches'] / max(1, original['close_matches'])):.1f}x")

if __name__ == "__main__":
    compare_with_original()