#!/usr/bin/env python3

import json
import numpy as np
import time
from ml_reimbursement_optimized import get_optimized_model
from ml_reimbursement_enhanced import get_enhanced_model

def compare_models():
    """Compare current optimized vs enhanced model"""
    print("🆚 MODEL COMPARISON - Current vs Enhanced")
    print("=" * 50)
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loading models...")
    current_model = get_optimized_model()
    enhanced_model = get_enhanced_model()
    
    print(f"Running comparison on {len(data)} cases...")
    
    # Run predictions for both models
    current_predictions = []
    enhanced_predictions = []
    actuals = []
    case_details = []
    
    start_time = time.time()
    
    for i, case in enumerate(data):
        if i % 200 == 0:
            print(f"Progress: {i}/{len(data)} cases...")
            
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        # Get predictions from both models
        current_pred = current_model.predict(days, miles, receipts)
        enhanced_pred = enhanced_model.predict(days, miles, receipts)
        
        current_predictions.append(current_pred)
        enhanced_predictions.append(enhanced_pred)
        actuals.append(expected)
        
        # Store case details
        efficiency = miles / max(days, 1)
        daily_spending = receipts / max(days, 1)
        
        case_details.append({
            'case_num': i + 1,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'efficiency': efficiency,
            'daily_spending': daily_spending,
            'expected': expected,
            'current_pred': current_pred,
            'enhanced_pred': enhanced_pred,
            'current_error': abs(current_pred - expected),
            'enhanced_error': abs(enhanced_pred - expected)
        })
    
    end_time = time.time()
    
    # Convert to numpy arrays
    current_predictions = np.array(current_predictions)
    enhanced_predictions = np.array(enhanced_predictions)
    actuals = np.array(actuals)
    
    current_errors = np.abs(current_predictions - actuals)
    enhanced_errors = np.abs(enhanced_predictions - actuals)
    
    # Calculate metrics for both models
    def calculate_metrics(errors, name):
        exact_matches = np.sum(errors < 0.01)
        very_close = np.sum(errors < 0.10)
        close_matches = np.sum(errors < 1.0)
        within_5 = np.sum(errors < 5.0)
        within_10 = np.sum(errors < 10.0)
        
        avg_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        return {
            'name': name,
            'exact_matches': exact_matches,
            'very_close': very_close,
            'close_matches': close_matches,
            'within_5': within_5,
            'within_10': within_10,
            'avg_error': avg_error,
            'median_error': median_error,
            'std_error': std_error,
            'max_error': max_error,
            'score': score
        }
    
    current_metrics = calculate_metrics(current_errors, "Current")
    enhanced_metrics = calculate_metrics(enhanced_errors, "Enhanced")
    
    # Display comparison
    print(f"\n📊 DETAILED COMPARISON RESULTS:")
    print(f"=" * 60)
    print(f"Total execution time: {end_time - start_time:.2f}s")
    print(f"")
    
    metrics_to_compare = [
        ('Exact matches (±$0.01)', 'exact_matches', 'higher'),
        ('Very close (±$0.10)', 'very_close', 'higher'),
        ('Close matches (±$1.00)', 'close_matches', 'higher'),
        ('Within $5.00', 'within_5', 'higher'),
        ('Within $10.00', 'within_10', 'higher'),
        ('Average error', 'avg_error', 'lower'),
        ('Median error', 'median_error', 'lower'),
        ('Std deviation', 'std_error', 'lower'),
        ('Max error', 'max_error', 'lower'),
        ('Score', 'score', 'lower')
    ]
    
    print(f"{'Metric':<20} {'Current':<12} {'Enhanced':<12} {'Winner':<10} {'Improvement'}")
    print("-" * 75)
    
    enhanced_wins = 0
    current_wins = 0
    
    for metric_name, metric_key, better in metrics_to_compare:
        current_val = current_metrics[metric_key]
        enhanced_val = enhanced_metrics[metric_key]
        
        if better == 'lower':
            winner = "Enhanced" if enhanced_val < current_val else "Current"
            if enhanced_val < current_val:
                improvement = f"{((current_val - enhanced_val)/current_val*100):.1f}% better"
                enhanced_wins += 1
            else:
                improvement = f"{((enhanced_val - current_val)/current_val*100):.1f}% worse"
                current_wins += 1
        else:
            winner = "Enhanced" if enhanced_val > current_val else "Current"
            if enhanced_val > current_val:
                improvement = f"+{enhanced_val - current_val} ({((enhanced_val - current_val)/max(current_val, 1)*100):.1f}%)"
                enhanced_wins += 1
            else:
                improvement = f"{enhanced_val - current_val} ({((current_val - enhanced_val)/max(current_val, 1)*100):.1f}%)"
                current_wins += 1
        
        if metric_name in ['Average error', 'Score']:
            print(f"{metric_name:<20} ${current_val:<11.2f} ${enhanced_val:<11.2f} {winner:<10} {improvement}")
        else:
            print(f"{metric_name:<20} {current_val:<12} {enhanced_val:<12} {winner:<10} {improvement}")
    
    print(f"\n🏆 OVERALL WINNER:")
    if enhanced_wins > current_wins:
        print(f"✅ ENHANCED MODEL WINS ({enhanced_wins}/{len(metrics_to_compare)} metrics)")
        recommendation = "DEPLOY ENHANCED VERSION"
    elif current_wins > enhanced_wins:
        print(f"❌ CURRENT MODEL WINS ({current_wins}/{len(metrics_to_compare)} metrics)")
        recommendation = "KEEP CURRENT VERSION"
    else:
        print(f"🤝 TIE ({enhanced_wins}/{len(metrics_to_compare)} metrics each)")
        # Tie-breaker: score and avg_error are most important
        if enhanced_metrics['score'] < current_metrics['score']:
            recommendation = "DEPLOY ENHANCED VERSION (better score)"
        else:
            recommendation = "KEEP CURRENT VERSION (better score)"
    
    # Analyze improvements in worst cases
    improvements_in_worst_cases = 0
    worst_case_improvements = []
    
    # Sort by current model's worst errors
    sorted_cases = sorted(case_details, key=lambda x: x['current_error'], reverse=True)
    
    print(f"\n💡 ANALYSIS OF TOP 10 WORST CASES:")
    print(f"=" * 50)
    for i, case in enumerate(sorted_cases[:10], 1):
        current_err = case['current_error']
        enhanced_err = case['enhanced_error']
        improvement = current_err - enhanced_err
        
        if improvement > 0:
            improvements_in_worst_cases += 1
            worst_case_improvements.append(improvement)
        
        print(f"{i:2d}. Case {case['case_num']:3d}: {case['days']}d, {case['miles']:4.0f}mi, ${case['receipts']:6.2f}")
        print(f"     Current: ${case['current_pred']:7.2f} (err: ${current_err:6.2f})")
        print(f"     Enhanced: ${case['enhanced_pred']:7.2f} (err: ${enhanced_err:6.2f})")
        if improvement > 0:
            print(f"     💚 IMPROVED by ${improvement:.2f}")
        elif improvement < 0:
            print(f"     🔴 WORSE by ${-improvement:.2f}")
        else:
            print(f"     ⚪ SAME")
        print()
    
    print(f"🎯 RECOMMENDATION: {recommendation}")
    print(f"📈 Worst cases improved: {improvements_in_worst_cases}/10")
    if worst_case_improvements:
        print(f"📈 Average improvement in worst cases: ${np.mean(worst_case_improvements):.2f}")
    
    # Final decision logic
    significant_improvement = (
        enhanced_metrics['score'] < current_metrics['score'] - 50 or  # Significant score improvement
        enhanced_metrics['avg_error'] < current_metrics['avg_error'] - 1 or  # $1+ error reduction
        enhanced_metrics['exact_matches'] > current_metrics['exact_matches'] + 2  # 2+ more exact matches
    )
    
    if enhanced_wins > current_wins and significant_improvement:
        final_decision = "DEPLOY ENHANCED VERSION - Significant improvement detected"
    elif enhanced_wins > current_wins:
        final_decision = "DEPLOY ENHANCED VERSION - Marginal improvement, but still better"
    else:
        final_decision = "KEEP CURRENT VERSION - Enhanced version not significantly better"
    
    print(f"\n🚀 FINAL DECISION: {final_decision}")
    
    return {
        'enhanced_better': enhanced_wins > current_wins,
        'significant_improvement': significant_improvement,
        'current_metrics': current_metrics,
        'enhanced_metrics': enhanced_metrics,
        'recommendation': final_decision
    }

if __name__ == "__main__":
    results = compare_models()