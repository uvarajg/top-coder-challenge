#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING THRESHOLD-BASED MILE RATES ===")

# Test if mile rate changes at certain thresholds
def calculate_reimbursement_threshold(days, miles, receipts):
    """Test threshold-based mile rates"""
    per_diem = days * 121.50
    
    # Test different threshold approaches
    if miles <= 500:
        mile_rate = 0.50
    elif miles <= 1000:
        mile_rate = 0.675
    else:
        mile_rate = 1.0  # Higher rate for 1000+ miles
    
    miles_component = miles * mile_rate
    receipts_component = receipts * 1.0
    
    return per_diem + miles_component + receipts_component

# Test another approach: different rates for different mile ranges
def calculate_reimbursement_tiered(days, miles, receipts):
    """Test tiered mile rates"""
    per_diem = days * 121.50
    
    # Tiered mile calculation
    miles_component = 0
    if miles <= 500:
        miles_component = miles * 0.50
    elif miles <= 1000:
        miles_component = 500 * 0.50 + (miles - 500) * 0.675
    else:
        miles_component = 500 * 0.50 + 500 * 0.675 + (miles - 1000) * 1.0
    
    receipts_component = receipts * 1.0
    
    return per_diem + miles_component + receipts_component

# Test the specific problem cases
target_cases = [148, 668, 512]

print("Testing threshold-based approaches:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    threshold_result = calculate_reimbursement_threshold(days, miles, receipts)
    tiered_result = calculate_reimbursement_tiered(days, miles, receipts)
    
    print(f"\nCase {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}")
    print(f"  Threshold approach: ${threshold_result:.2f} (error: ${abs(threshold_result - expected):.2f})")
    print(f"  Tiered approach: ${tiered_result:.2f} (error: ${abs(tiered_result - expected):.2f})")

# Now test a hypothesis: what if the coefficients in the additive formula depend on the values?
print("\n=== TESTING DYNAMIC COEFFICIENT HYPOTHESIS ===")

# Looking at the analysis above, it seems like for high-mile cases:
# The receipt multiplier decreases as mile rate increases
# Let's see if there's a pattern

def test_dynamic_formula(days, miles, receipts):
    """Test if coefficients change based on input values"""
    per_diem = days * 121.50
    
    # Hypothesis: mile rate increases with total miles, receipt rate decreases
    if miles < 500:
        mile_rate = 0.50
        receipt_rate = 1.0
    elif miles < 1000:
        mile_rate = 0.675
        receipt_rate = 1.0
    else:
        # For 1000+ miles, higher mile rate but lower receipt rate?
        mile_rate = 1.0
        receipt_rate = 0.5  # Half receipt reimbursement for high-mile trips?
    
    return per_diem + miles * mile_rate + receipts * receipt_rate

print("Testing dynamic coefficient approach:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    dynamic_result = test_dynamic_formula(days, miles, receipts)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Dynamic: ${dynamic_result:.2f}, Error: ${abs(dynamic_result - expected):.2f}")

# Test this on more high-mile cases
print("\n=== TESTING DYNAMIC FORMULA ON MORE HIGH-MILE CASES ===")
high_mile_cases = [(i, case) for i, case in enumerate(cases) if case['input']['miles_traveled'] >= 1000]

errors = []
for case_id, case in high_mile_cases[:20]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = test_dynamic_formula(days, miles, receipts)
    error = abs(predicted - expected)
    errors.append(error)
    
    if error < 100:  # Show cases with low error
        print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

avg_error = sum(errors) / len(errors)
print(f"\nAverage error on 20 high-mile cases: ${avg_error:.2f}")
print(f"Max error: ${max(errors):.2f}")
print(f"Cases with error < $100: {sum(1 for e in errors if e < 100)}/{len(errors)}")

# Let's also test if the pattern holds for medium-mile cases
print("\n=== TESTING ON MEDIUM-MILE CASES (500-1000) ===")
medium_mile_cases = [(i, case) for i, case in enumerate(cases) if 500 <= case['input']['miles_traveled'] < 1000]

med_errors = []
for case_id, case in medium_mile_cases[:10]:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = test_dynamic_formula(days, miles, receipts)
    error = abs(predicted - expected)
    med_errors.append(error)
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

med_avg_error = sum(med_errors) / len(med_errors)
print(f"\nAverage error on medium-mile cases: ${med_avg_error:.2f}")