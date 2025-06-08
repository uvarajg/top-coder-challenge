#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== DEEP DIVE INTO MAX FORMULA HYPOTHESIS ===")

# Based on our earlier findings, let's test: per_diem + max(miles*rate, receipts*rate)
# The max formula showed better results. Let's refine it.

def test_max_formula_variations(days, miles, receipts):
    """Test different variations of max formula"""
    per_diem = days * 121.50
    
    results = {}
    
    # Variation 1: max(miles*0.675, receipts*1.0)
    results['v1'] = per_diem + max(miles * 0.675, receipts * 1.0)
    
    # Variation 2: max(miles*0.50, receipts*1.0) 
    results['v2'] = per_diem + max(miles * 0.50, receipts * 1.0)
    
    # Variation 3: max(miles*1.0, receipts*1.0)
    results['v3'] = per_diem + max(miles * 1.0, receipts * 1.0)
    
    # Variation 4: max(miles*0.675, receipts*1.5)
    results['v4'] = per_diem + max(miles * 0.675, receipts * 1.5)
    
    # Variation 5: What if there's a base mile component plus max?
    # base_miles + max(extra_miles*rate, receipts*rate)
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    results['v5'] = per_diem + base_miles + max(extra_miles, receipts * 1.0)
    
    return results

# Test on our problem cases
target_cases = [148, 668, 512]

print("Testing max formula variations on problem cases:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    results = test_max_formula_variations(days, miles, receipts)
    
    print(f"\nCase {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}")
    for variant, result in results.items():
        error = abs(result - expected)
        print(f"  {variant}: ${result:.2f} (error: ${error:.2f})")

# Test the best performing variation on more cases
print("\n=== TESTING BEST VARIATION ON MORE CASES ===")

# Find high-mile cases to test
high_mile_cases = [(i, case) for i, case in enumerate(cases) if case['input']['miles_traveled'] >= 1000]

# Test variation 1 (max(miles*0.675, receipts*1.0)) on more cases
errors_v1 = []
good_cases_v1 = []

for case_id, case in high_mile_cases[:30]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem = days * 121.50
    predicted = per_diem + max(miles * 0.675, receipts * 1.0)
    error = abs(predicted - expected)
    errors_v1.append(error)
    
    if error < 200:  # Cases with reasonable error
        good_cases_v1.append((case_id, error))

print(f"Variation 1 on 30 high-mile cases:")
print(f"  Average error: ${sum(errors_v1)/len(errors_v1):.2f}")
print(f"  Max error: ${max(errors_v1):.2f}")
print(f"  Cases with error < $200: {len(good_cases_v1)}/30")

# Show the good cases
print(f"\nCases with error < $200:")
for case_id, error in good_cases_v1[:10]:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    per_diem = days * 121.50
    predicted = per_diem + max(miles * 0.675, receipts * 1.0)
    
    print(f"  Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")
    
    # Show which component was chosen by max
    miles_comp = miles * 0.675
    receipts_comp = receipts * 1.0
    chosen = "miles" if miles_comp > receipts_comp else "receipts"
    print(f"    Max chose {chosen} (${miles_comp:.2f} vs ${receipts_comp:.2f})")

# Let's also test on some medium and low mile cases to see if this breaks them
print(f"\n=== TESTING ON MEDIUM/LOW MILE CASES ===")

medium_cases = [(i, case) for i, case in enumerate(cases) if 200 <= case['input']['miles_traveled'] < 1000]
med_errors = []

for case_id, case in medium_cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem = days * 121.50
    predicted = per_diem + max(miles * 0.675, receipts * 1.0)
    error = abs(predicted - expected)
    med_errors.append(error)
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

med_avg_error = sum(med_errors) / len(med_errors)
print(f"\nMedium mile cases average error: ${med_avg_error:.2f}")

# Test on low mile cases
low_cases = [(i, case) for i, case in enumerate(cases) if case['input']['miles_traveled'] < 200]
low_errors = []

for case_id, case in low_cases[:5]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem = days * 121.50
    predicted = per_diem + max(miles * 0.675, receipts * 1.0)
    error = abs(predicted - expected)
    low_errors.append(error)
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

if low_errors:
    low_avg_error = sum(low_errors) / len(low_errors)
    print(f"\nLow mile cases average error: ${low_avg_error:.2f}")