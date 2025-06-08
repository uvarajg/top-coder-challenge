import json
import numpy as np

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Extract data
trip_days = []
miles = []
receipts = []
reimbursements = []

for case in data:
    trip_days.append(case['input']['trip_duration_days'])
    miles.append(case['input']['miles_traveled'])
    receipts.append(case['input']['total_receipts_amount'])
    reimbursements.append(case['expected_output'])

trip_days = np.array(trip_days)
miles = np.array(miles)
receipts = np.array(receipts)
reimbursements = np.array(reimbursements)

print('=== FINAL ANALYSIS: LOOKING FOR THE EXACT FORMULA ===')

# Test hypothesis: Different rates based on trip length or efficiency
miles_per_day = miles / trip_days

# Group by trip length
short_trips = trip_days <= 3
medium_trips = (trip_days > 3) & (trip_days <= 7)
long_trips = trip_days > 7

print('\nAnalyzing by trip length:')
for name, mask in [('Short (1-3 days)', short_trips), 
                   ('Medium (4-7 days)', medium_trips), 
                   ('Long (8+ days)', long_trips)]:
    if np.sum(mask) > 50:
        subset_reimb = reimbursements[mask]
        subset_days = trip_days[mask]
        subset_miles = miles[mask]
        subset_receipts = receipts[mask]
        subset_mpd = miles_per_day[mask]
        
        print(f'\n{name} ({np.sum(mask)} cases):')
        print(f'  Avg reimbursement/day: ${(subset_reimb/subset_days).mean():.2f}')
        print(f'  Avg reimbursement/mile: ${(subset_reimb/subset_miles).mean():.3f}')
        print(f'  Avg miles/day: {subset_mpd.mean():.1f}')
        
        # Test linear fit for this subset
        from scipy.optimize import minimize
        
        def subset_formula(params):
            base, mile_rate, receipt_mult = params
            predicted = base * subset_days + mile_rate * subset_miles + receipt_mult * subset_receipts
            return np.sum((predicted - subset_reimb) ** 2)
        
        result = minimize(subset_formula, [100, 0.56, 0.5])
        base, mile_rate, receipt_mult = result.x
        
        predicted = base * subset_days + mile_rate * subset_miles + receipt_mult * subset_receipts
        rmse = np.sqrt(np.mean((predicted - subset_reimb) ** 2))
        
        print(f'  Best fit: ${base:.2f}/day + ${mile_rate:.3f}/mile + {receipt_mult:.3f}*receipts')
        print(f'  RMSE: {rmse:.2f}')
        
        # Check for exact matches
        exact = np.sum(np.abs(predicted - subset_reimb) < 0.01)
        print(f'  Exact matches: {exact}')

print('\n=== TESTING PIECEWISE MODELS ===')

# Test if there are different formulas for different efficiency ranges
def test_piecewise_efficiency():
    # Low efficiency: < 50 miles/day
    # High efficiency: >= 50 miles/day
    
    low_eff = miles_per_day < 50
    high_eff = miles_per_day >= 50
    
    print(f'\nLow efficiency (<50 mi/day): {np.sum(low_eff)} cases')
    print(f'High efficiency (>=50 mi/day): {np.sum(high_eff)} cases')
    
    total_error = 0
    
    for name, mask in [('Low efficiency', low_eff), ('High efficiency', high_eff)]:
        if np.sum(mask) > 50:
            subset_reimb = reimbursements[mask]
            subset_days = trip_days[mask]
            subset_miles = miles[mask] 
            subset_receipts = receipts[mask]
            
            # Fit formula for this efficiency group
            from scipy.optimize import minimize
            
            def eff_formula(params):
                base, mile_rate, receipt_mult = params
                predicted = base * subset_days + mile_rate * subset_miles + receipt_mult * subset_receipts
                return np.sum((predicted - subset_reimb) ** 2)
            
            result = minimize(eff_formula, [100, 0.56, 0.5])
            base, mile_rate, receipt_mult = result.x
            
            predicted = base * subset_days + mile_rate * subset_miles + receipt_mult * subset_receipts
            rmse = np.sqrt(np.mean((predicted - subset_reimb) ** 2))
            total_error += np.sum((predicted - subset_reimb) ** 2)
            
            print(f'{name}: ${base:.2f}/day + ${mile_rate:.3f}/mile + {receipt_mult:.3f}*receipts, RMSE: {rmse:.2f}')
    
    overall_rmse = np.sqrt(total_error / len(reimbursements))
    print(f'Overall piecewise RMSE: {overall_rmse:.2f}')

test_piecewise_efficiency()

print('\n=== SUMMARY OF FINDINGS ===')
print('1. Strong correlation between receipts and reimbursement (r=0.704)')
print('2. Moderate correlations with days (r=0.514) and miles (r=0.432)')
print('3. Best simple formula: ~$63/day + $0.58/mile + 0.43*receipts')
print('4. High efficiency trips seem to have different rates')
print('5. No exact pattern matches found - suggests complex legacy calculation')
print('6. Large residuals indicate potential caps, bonuses, or conditional logic')
print('7. Per-day rates decrease with trip length (efficiency effect)')

# Final recommendation
print('\n=== RECOMMENDED APPROACH ===')
print('Given the complexity and lack of exact matches, the legacy system likely uses:')
print('- Base per diem rate around $60-75/day')
print('- Standard mileage rate around $0.56-0.58/mile') 
print('- Partial receipt reimbursement at ~40-50%')
print('- Efficiency adjustments based on miles/day')
print('- Possible caps or special handling for long trips')
print('- Legacy rounding or calculation quirks causing inexact matches')