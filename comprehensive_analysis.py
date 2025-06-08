#!/usr/bin/env python3

import json
import numpy as np
import time
from ml_reimbursement_optimized import get_optimized_model

def analyze_current_system():
    """Comprehensive analysis of current ML system"""
    print("🔍 COMPREHENSIVE ANALYSIS - Current ML System")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loading optimized ML model...")
    model = get_optimized_model()
    
    # Run all 1000 predictions
    predictions = []
    actuals = []
    errors = []
    case_details = []
    
    print(f"Running {len(data)} predictions...")
    
    for i, case in enumerate(data):
        if i % 200 == 0:
            print(f"Progress: {i}/{len(data)} cases...")
            
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        prediction = model.predict(days, miles, receipts)
        error = abs(prediction - expected)
        
        predictions.append(prediction)
        actuals.append(expected)
        errors.append(error)
        
        # Store case details for analysis
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
            'predicted': prediction,
            'error': error
        })
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    # Calculate comprehensive metrics
    exact_matches = np.sum(errors < 0.01)
    very_close = np.sum(errors < 0.10)
    close_matches = np.sum(errors < 1.0)
    within_5 = np.sum(errors < 5.0)
    within_10 = np.sum(errors < 10.0)
    
    avg_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    end_time = time.time()
    
    print(f"\n📊 COMPREHENSIVE RESULTS:")
    print(f"=" * 40)
    print(f"Execution time: {end_time - start_time:.2f}s")
    print(f"Total cases: {len(data)}")
    print(f"")
    print(f"🎯 Accuracy Metrics:")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/1000*100:.2f}%)")
    print(f"  Very close (±$0.10): {very_close} ({very_close/1000*100:.2f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/1000*100:.2f}%)")
    print(f"  Within $5.00: {within_5} ({within_5/1000*100:.2f}%)")
    print(f"  Within $10.00: {within_10} ({within_10/1000*100:.2f}%)")
    print(f"")
    print(f"📈 Error Statistics:")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Median error: ${median_error:.2f}")
    print(f"  Std deviation: ${std_error:.2f}")
    print(f"  Min error: ${min_error:.2f}")
    print(f"  Max error: ${max_error:.2f}")
    print(f"  Score: {score:.2f}")
    
    # Analyze error patterns
    print(f"\n🔍 ERROR PATTERN ANALYSIS:")
    print(f"=" * 30)
    
    # Group by trip characteristics
    short_trips = [d for d in case_details if d['days'] <= 2]
    medium_trips = [d for d in case_details if 3 <= d['days'] <= 7]
    long_trips = [d for d in case_details if d['days'] >= 8]
    
    print(f"Trip Length Analysis:")
    print(f"  Short trips (≤2 days): {len(short_trips)} cases, avg error: ${np.mean([d['error'] for d in short_trips]):.2f}")
    print(f"  Medium trips (3-7 days): {len(medium_trips)} cases, avg error: ${np.mean([d['error'] for d in medium_trips]):.2f}")
    print(f"  Long trips (≥8 days): {len(long_trips)} cases, avg error: ${np.mean([d['error'] for d in long_trips]):.2f}")
    
    # Group by efficiency
    low_eff = [d for d in case_details if d['efficiency'] < 50]
    med_eff = [d for d in case_details if 50 <= d['efficiency'] < 150]
    high_eff = [d for d in case_details if d['efficiency'] >= 150]
    
    print(f"\nEfficiency Analysis:")
    print(f"  Low efficiency (<50): {len(low_eff)} cases, avg error: ${np.mean([d['error'] for d in low_eff]):.2f}")
    print(f"  Medium efficiency (50-150): {len(med_eff)} cases, avg error: ${np.mean([d['error'] for d in med_eff]):.2f}")
    print(f"  High efficiency (≥150): {len(high_eff)} cases, avg error: ${np.mean([d['error'] for d in high_eff]):.2f}")
    
    # Group by receipt amount
    low_rec = [d for d in case_details if d['receipts'] < 500]
    med_rec = [d for d in case_details if 500 <= d['receipts'] < 1500]
    high_rec = [d for d in case_details if d['receipts'] >= 1500]
    
    print(f"\nReceipt Amount Analysis:")
    print(f"  Low receipts (<$500): {len(low_rec)} cases, avg error: ${np.mean([d['error'] for d in low_rec]):.2f}")
    print(f"  Medium receipts ($500-1500): {len(med_rec)} cases, avg error: ${np.mean([d['error'] for d in med_rec]):.2f}")
    print(f"  High receipts (≥$1500): {len(high_rec)} cases, avg error: ${np.mean([d['error'] for d in high_rec]):.2f}")
    
    # Show worst cases for improvement opportunities
    print(f"\n💡 TOP 10 WORST CASES FOR OPTIMIZATION:")
    print(f"=" * 45)
    worst_cases = sorted(case_details, key=lambda x: x['error'], reverse=True)[:10]
    
    for i, case in enumerate(worst_cases, 1):
        print(f"{i:2d}. Case {case['case_num']:3d}: {case['days']}d, {case['miles']:4.0f}mi, ${case['receipts']:6.2f}")
        print(f"     Eff: {case['efficiency']:5.1f}, Daily: ${case['daily_spending']:5.2f}")
        print(f"     Expected: ${case['expected']:7.2f}, Got: ${case['predicted']:7.2f}, Error: ${case['error']:6.2f}")
        print()
    
    return {
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'avg_error': avg_error,
        'score': score,
        'execution_time': end_time - start_time,
        'case_details': case_details,
        'worst_cases': worst_cases
    }

if __name__ == "__main__":
    results = analyze_current_system()
    print(f"\n🎯 BASELINE ESTABLISHED:")
    print(f"  Score to beat: {results['score']:.2f}")
    print(f"  Avg error to beat: ${results['avg_error']:.2f}")
    print(f"  Exact matches to beat: {results['exact_matches']}")