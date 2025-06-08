#!/usr/bin/env python3

import json
import numpy as np
from advanced_reimbursement_system import PracticalReimbursementSystem

def evaluate_advanced_system():
    """Comprehensive evaluation of the advanced business rule engine"""
    
    print("🧾 Advanced Business Rule Engine - Comprehensive Evaluation")
    print("=========================================================")
    print()
    
    # Train the advanced system
    print("📊 Training advanced multi-layered business rule engine...")
    system = PracticalReimbursementSystem()
    X, y = system.load_data()
    results = system.train_models(X, y)
    
    print()
    print("📊 Running detailed evaluation...")
    print()
    
    # Detailed evaluation on all cases
    predictions = []
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    very_close_matches = 0  # ±$0.10
    total_error = 0.0
    max_error = 0.0
    max_error_case = ""
    errors_list = []
    
    for i, (features, expected) in enumerate(zip(X, y)):
        days, miles, receipts = features
        
        try:
            prediction = system.predict(days, miles, receipts)
            error = abs(prediction - expected)
            
            predictions.append(prediction)
            successful_runs += 1
            
            # Check for exact match (within $0.01)
            if error < 0.01:
                exact_matches += 1
            
            # Check for very close match (within $0.10)
            if error < 0.10:
                very_close_matches += 1
            
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
        very_close_pct = very_close_matches * 100 / successful_runs
        close_pct = close_matches * 100 / successful_runs
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        print("✅ Advanced System Evaluation Complete!")
        print("")
        print("📈 Performance Results:")
        print(f"  Total test cases: 1000")
        print(f"  Successful runs: {successful_runs}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
        print(f"  Very close matches (±$0.10): {very_close_matches} ({very_close_pct:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Maximum error: ${max_error:.2f}")
        print("")
        print(f"🎯 Score: {score:.2f} (lower is better)")
        print("")
        
        # Performance analysis by trip type
        print("📊 Performance Analysis by Trip Characteristics:")
        print("")
        
        # Analyze by trip length
        short_trips = [(e, p, err) for _, e, p, err, d, m, r in errors_list if d <= 3]
        medium_trips = [(e, p, err) for _, e, p, err, d, m, r in errors_list if 4 <= d <= 7]
        long_trips = [(e, p, err) for _, e, p, err, d, m, r in errors_list if d >= 8]
        
        print(f"  Short trips (≤3 days): {len(short_trips)} cases, avg error: ${np.mean([err for _, _, err in short_trips]):.2f}")
        print(f"  Medium trips (4-7 days): {len(medium_trips)} cases, avg error: ${np.mean([err for _, _, err in medium_trips]):.2f}")
        print(f"  Long trips (≥8 days): {len(long_trips)} cases, avg error: ${np.mean([err for _, _, err in long_trips]):.2f}")
        print("")
        
        # Analyze by efficiency
        low_eff = [(e, p, err) for _, e, p, err, d, m, r in errors_list if (m/d) < 50]
        med_eff = [(e, p, err) for _, e, p, err, d, m, r in errors_list if 50 <= (m/d) < 150]
        high_eff = [(e, p, err) for _, e, p, err, d, m, r in errors_list if (m/d) >= 150]
        
        print(f"  Low efficiency (<50 mi/day): {len(low_eff)} cases, avg error: ${np.mean([err for _, _, err in low_eff]):.2f}")
        print(f"  Medium efficiency (50-150 mi/day): {len(med_eff)} cases, avg error: ${np.mean([err for _, _, err in med_eff]):.2f}")
        print(f"  High efficiency (≥150 mi/day): {len(high_eff)} cases, avg error: ${np.mean([err for _, _, err in high_eff]):.2f}")
        print("")
        
        # Analyze by receipt amount
        low_rec = [(e, p, err) for _, e, p, err, d, m, r in errors_list if r < 500]
        med_rec = [(e, p, err) for _, e, p, err, d, m, r in errors_list if 500 <= r < 1500]
        high_rec = [(e, p, err) for _, e, p, err, d, m, r in errors_list if r >= 1500]
        
        print(f"  Low receipts (<$500): {len(low_rec)} cases, avg error: ${np.mean([err for _, _, err in low_rec]):.2f}")
        print(f"  Medium receipts ($500-1500): {len(med_rec)} cases, avg error: ${np.mean([err for _, _, err in med_rec]):.2f}")
        print(f"  High receipts (≥$1500): {len(high_rec)} cases, avg error: ${np.mean([err for _, _, err in high_rec]):.2f}")
        print("")
        
        # Provide feedback based on exact matches
        if exact_matches >= 100:
            print("🏆 OUTSTANDING! You have achieved significant exact matches!")
        elif exact_matches >= 50:
            print("🥇 Excellent! Very strong performance on exact matches.")
        elif exact_matches >= 20:
            print("🥈 Great work! Good progress on exact matches.")
        elif exact_matches >= 5:
            print("🥉 Good start! Some exact matches achieved.")
        else:
            print("📚 Continue optimizing for exact matches.")
        
        print("")
        print("💡 Advanced System Analysis:")
        if exact_matches < 50:
            print("  Top error cases for pattern analysis:")
            
            # Sort by error and show top 5
            errors_list.sort(key=lambda x: x[3], reverse=True)
            for i in range(min(5, len(errors_list))):
                case_num, expected, actual, error, days, miles, receipts = errors_list[i]
                efficiency = miles / days
                daily_spending = receipts / days
                print(f"    Case {case_num}: {days}d, {miles}mi, ${receipts:.2f} (eff: {efficiency:.1f}, daily: ${daily_spending:.2f})")
                print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
        
        return {
            'successful_runs': successful_runs,
            'exact_matches': exact_matches,
            'very_close_matches': very_close_matches,
            'close_matches': close_matches,
            'avg_error': avg_error,
            'max_error': max_error,
            'score': score,
            'predictions': predictions
        }
    else:
        print("❌ No successful test cases!")
        return None

def compare_all_systems():
    """Compare original, ML, and advanced systems"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE SYSTEM COMPARISON")
    print("="*70)
    
    # Original performance (from your eval.sh results)
    original = {
        'exact_matches': 1,
        'close_matches': 7,
        'avg_error': 183.13,
        'score': 18412.90
    }
    
    # ML performance (from previous results)
    ml_system = {
        'exact_matches': 0,
        'close_matches': 23,
        'avg_error': 27.61,
        'score': 2860.55
    }
    
    # Advanced system performance
    advanced_results = evaluate_advanced_system()
    
    if advanced_results:
        print(f"\n📈 EVOLUTION OF PERFORMANCE:")
        print(f"{'Metric':<25} {'Original':<12} {'ML System':<12} {'Advanced':<12} {'Best Improvement'}")
        print("-" * 70)
        print(f"{'Exact matches':<25} {original['exact_matches']:<12} {ml_system['exact_matches']:<12} {advanced_results['exact_matches']:<12} {advanced_results['exact_matches']}x better")
        print(f"{'Close matches':<25} {original['close_matches']:<12} {ml_system['close_matches']:<12} {advanced_results['close_matches']:<12} {advanced_results['close_matches']/max(1,original['close_matches']):.1f}x better")
        print(f"{'Average error':<25} ${original['avg_error']:<11.2f} ${ml_system['avg_error']:<11.2f} ${advanced_results['avg_error']:<11.2f} {((original['avg_error'] - advanced_results['avg_error'])/original['avg_error']*100):.1f}% reduction")
        print(f"{'Score':<25} {original['score']:<12.2f} {ml_system['score']:<12.2f} {advanced_results['score']:<12.2f} {((original['score'] - advanced_results['score'])/original['score']*100):.1f}% better")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        if advanced_results['exact_matches'] > ml_system['exact_matches']:
            print(f"✅ Advanced system achieved {advanced_results['exact_matches']} exact matches vs {ml_system['exact_matches']} from ML system")
        if advanced_results['avg_error'] < ml_system['avg_error']:
            improvement = ((ml_system['avg_error'] - advanced_results['avg_error'])/ml_system['avg_error']*100)
            print(f"✅ Advanced system reduced error by {improvement:.1f}% vs ML system")
        if advanced_results['score'] < ml_system['score']:
            improvement = ((ml_system['score'] - advanced_results['score'])/ml_system['score']*100)
            print(f"✅ Advanced system improved score by {improvement:.1f}% vs ML system")

if __name__ == "__main__":
    compare_all_systems()