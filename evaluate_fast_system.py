#!/usr/bin/env python3

import json
import numpy as np
from fast_advanced_system import FastBusinessRuleEngine

def evaluate_fast_system():
    """Evaluate the fast advanced system"""
    
    print("🧾 Fast Advanced Business Rule Engine - Evaluation")
    print("=================================================")
    print()
    
    # Train system
    system = FastBusinessRuleEngine()
    X, y = system.load_data()
    results = system.train(X, y)
    
    print()
    print("📊 Detailed Performance Analysis:")
    print()
    
    # Detailed evaluation
    predictions = []
    exact_matches = 0
    very_close_matches = 0  # ±$0.10
    close_matches = 0
    total_error = 0.0
    max_error = 0.0
    errors_list = []
    
    for i, (features, expected) in enumerate(zip(X, y)):
        days, miles, receipts = features
        prediction = system.predict(days, miles, receipts)
        error = abs(prediction - expected)
        
        predictions.append(prediction)
        
        if error < 0.01:
            exact_matches += 1
        if error < 0.10:
            very_close_matches += 1
        if error < 1.0:
            close_matches += 1
        
        total_error += error
        if error > max_error:
            max_error = error
        
        errors_list.append((i+1, expected, prediction, error, days, miles, receipts))
    
    avg_error = total_error / len(y)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"✅ Fast Advanced System Results:")
    print(f"  Total cases: 1000")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/10:.1f}%)")
    print(f"  Very close matches (±$0.10): {very_close_matches} ({very_close_matches/10:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/10:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Score: {score:.2f}")
    print()
    
    # Performance by category
    short_trips = [err for _, _, _, err, d, _, _ in errors_list if d <= 3]
    medium_trips = [err for _, _, _, err, d, _, _ in errors_list if 4 <= d <= 7]
    long_trips = [err for _, _, _, err, d, _, _ in errors_list if d >= 8]
    
    print(f"📊 Performance by Trip Length:")
    print(f"  Short trips (≤3 days): {len(short_trips)} cases, avg error: ${np.mean(short_trips):.2f}")
    print(f"  Medium trips (4-7 days): {len(medium_trips)} cases, avg error: ${np.mean(medium_trips):.2f}")
    print(f"  Long trips (≥8 days): {len(long_trips)} cases, avg error: ${np.mean(long_trips):.2f}")
    print()
    
    # Show improvement insights
    if exact_matches >= 10:
        print(f"🏆 EXCELLENT! Achieved {exact_matches} exact matches!")
    elif exact_matches >= 5:
        print(f"🥇 GREAT! Achieved {exact_matches} exact matches!")
    elif exact_matches >= 1:
        print(f"🥈 GOOD! Achieved {exact_matches} exact matches!")
    else:
        print("📈 Continue optimizing for exact matches")
    
    # Show worst cases for analysis
    if exact_matches < 20:
        print()
        print("💡 Top error cases for further optimization:")
        errors_list.sort(key=lambda x: x[3], reverse=True)
        for i in range(min(5, len(errors_list))):
            case_num, expected, actual, error, days, miles, receipts = errors_list[i]
            efficiency = miles / days
            daily_spending = receipts / days
            print(f"  Case {case_num}: {days}d, {miles}mi, ${receipts:.2f}")
            print(f"    Efficiency: {efficiency:.1f} mi/day, Daily spending: ${daily_spending:.2f}")
            print(f"    Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
    
    return {
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'avg_error': avg_error,
        'score': score
    }

def compare_all_systems():
    """Compare all system versions"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE SYSTEM COMPARISON")
    print("="*70)
    
    # Historical performance
    original = {'exact_matches': 1, 'close_matches': 7, 'avg_error': 183.13, 'score': 18412.90}
    ml_system = {'exact_matches': 0, 'close_matches': 23, 'avg_error': 27.61, 'score': 2860.55}
    
    # Fast advanced system
    fast_results = evaluate_fast_system()
    
    print(f"\n📈 SYSTEM EVOLUTION:")
    print(f"{'Metric':<20} {'Original':<10} {'ML':<10} {'Advanced':<10} {'Best'}")
    print("-" * 60)
    print(f"{'Exact matches':<20} {original['exact_matches']:<10} {ml_system['exact_matches']:<10} {fast_results['exact_matches']:<10} {max(original['exact_matches'], ml_system['exact_matches'], fast_results['exact_matches'])}")
    print(f"{'Close matches':<20} {original['close_matches']:<10} {ml_system['close_matches']:<10} {fast_results['close_matches']:<10} {max(original['close_matches'], ml_system['close_matches'], fast_results['close_matches'])}")
    print(f"{'Avg Error':<20} ${original['avg_error']:<9.2f} ${ml_system['avg_error']:<9.2f} ${fast_results['avg_error']:<9.2f} ${min(original['avg_error'], ml_system['avg_error'], fast_results['avg_error']):.2f}")
    print(f"{'Score':<20} {original['score']:<10.0f} {ml_system['score']:<10.0f} {fast_results['score']:<10.0f} {min(original['score'], ml_system['score'], fast_results['score']):.0f}")
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    best_exact = max(original['exact_matches'], ml_system['exact_matches'], fast_results['exact_matches'])
    best_error = min(original['avg_error'], ml_system['avg_error'], fast_results['avg_error'])
    
    if fast_results['exact_matches'] == best_exact:
        print(f"✅ Advanced system achieved best exact matches: {fast_results['exact_matches']}")
    if abs(fast_results['avg_error'] - best_error) < 0.01:
        print(f"✅ Advanced system achieved best average error: ${fast_results['avg_error']:.2f}")
    
    # Calculate overall improvement
    error_improvement = ((original['avg_error'] - fast_results['avg_error'])/original['avg_error']*100)
    score_improvement = ((original['score'] - fast_results['score'])/original['score']*100)
    
    print(f"\n🚀 OVERALL IMPROVEMENT FROM ORIGINAL:")
    print(f"  Average error: {error_improvement:.1f}% reduction")
    print(f"  Score: {score_improvement:.1f}% improvement")
    print(f"  Exact matches: {fast_results['exact_matches']}x increase")

if __name__ == "__main__":
    compare_all_systems()