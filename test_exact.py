import json
import numpy as np

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Extract first 50 cases for detailed analysis
cases = data[:50]

print('=== DETAILED ANALYSIS OF FIRST 50 CASES ===')
print('Case | Days | Miles | Receipts | Reimbursement | Miles/Day | Reimb/Day | Reimb/Mile')
print('-' * 85)

for i, case in enumerate(cases):
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    reimb = case['expected_output']
    
    mpd = miles / days
    rpd = reimb / days
    rpm = reimb / miles
    
    print(f'{i:4d} | {days:4d} | {miles:5.0f} | {receipts:8.2f} | {reimb:11.2f} | {mpd:8.1f} | {rpd:8.2f} | {rpm:9.3f}')

print('\n=== TESTING SPECIFIC RATE COMBINATIONS ===')

# Test common government rates
test_combinations = [
    (75, 0.56, 0.5),   # Common federal rates
    (50, 0.56, 1.0),   # Full receipt reimbursement
    (100, 0.58, 0.3),  # Higher per diem, newer mileage rate
    (60, 0.50, 0.75),  # Round numbers
]

for base_day, mile_rate, receipt_mult in test_combinations:
    exact_matches = 0
    close_matches = 0
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        actual = case['expected_output']
        
        predicted = base_day * days + mile_rate * miles + receipt_mult * receipts
        
        if abs(predicted - actual) < 0.01:
            exact_matches += 1
        elif abs(predicted - actual) < 1.0:
            close_matches += 1
    
    print(f'${base_day}/day + ${mile_rate}/mile + {receipt_mult}*receipts: {exact_matches} exact, {close_matches} close matches')

print('\n=== LOOKING FOR PATTERNS IN DIFFERENCES ===')

# Test the best formula found and analyze residuals
best_formula = lambda d, m, r: 63.19 * d + 0.578 * m + 0.434 * r

residuals = []
for case in data:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    actual = case['expected_output']
    
    predicted = best_formula(days, miles, receipts)
    residual = actual - predicted
    residuals.append(residual)

residuals = np.array(residuals)

print(f'Residual statistics:')
print(f'  Mean: {residuals.mean():.3f}')
print(f'  Std: {residuals.std():.3f}')
print(f'  Min: {residuals.min():.3f}')
print(f'  Max: {residuals.max():.3f}')

# Look for patterns in residuals
print(f'\nLarge positive residuals (actual > predicted by >$100):')
large_pos = residuals > 100
if np.sum(large_pos) > 0:
    for i, case in enumerate(data):
        if large_pos[i]:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            actual = case['expected_output']
            mpd = miles / days
            print(f'  {days} days, {miles:.0f} miles, ${receipts:.2f} receipts, ${actual:.2f} actual, {mpd:.1f} mi/day, residual: ${residuals[i]:.2f}')
            if np.sum(large_pos) > 10:  # Limit output
                break

print(f'\nLarge negative residuals (actual < predicted by >$100):')
large_neg = residuals < -100
if np.sum(large_neg) > 0:
    for i, case in enumerate(data):
        if large_neg[i]:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            actual = case['expected_output']
            mpd = miles / days
            print(f'  {days} days, {miles:.0f} miles, ${receipts:.2f} receipts, ${actual:.2f} actual, {mpd:.1f} mi/day, residual: ${residuals[i]:.2f}')
            if np.sum(large_neg) > 10:  # Limit output
                break