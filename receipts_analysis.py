#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING RECEIPT PATTERNS ===")

# Look at the worst cases from our formula and see what the receipt pattern is
worst_case_ids = [585, 241, 150, 151, 493, 250, 317, 919, 406, 644]

print("Worst cases analysis:")
for case_id in worst_case_ids:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    # Calculate what receipt rate would be needed if we use our base formula
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    
    # If we take max of extra_miles vs receipts, and receipts wins
    if receipts > extra_miles:
        # Then: per_diem + base_miles + receipts*rate = expected
        remaining = expected - per_diem - base_miles
        needed_receipt_rate = remaining / receipts if receipts > 0 else 0
    else:
        # Extra miles wins, so receipts don't contribute
        needed_receipt_rate = 0
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}")
    print(f"  Per diem: ${per_diem:.2f}, Base miles: ${base_miles:.2f}, Extra miles: ${extra_miles:.2f}")
    print(f"  Needed receipt rate: {needed_receipt_rate:.3f}")

# Let's see if there's a receipt cap or different receipt rate
print(f"\n=== TESTING RECEIPT CAP HYPOTHESIS ===")

def calculate_with_receipt_cap(days, miles, receipts, receipt_cap=1000):
    """Test formula with receipt cap"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    capped_receipts = min(receipts, receipt_cap)
    
    return per_diem + base_miles + max(extra_miles, capped_receipts)

# Test different receipt caps
caps_to_test = [500, 750, 1000, 1250, 1500]

for cap in caps_to_test:
    errors = []
    for case_id in worst_case_ids:
        case = cases[case_id]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_with_receipt_cap(days, miles, receipts, cap)
        error = abs(predicted - expected)
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    print(f"Receipt cap ${cap}: Average error on worst cases: ${avg_error:.2f}")

# Test a different hypothesis: maybe receipts are only partially reimbursed
print(f"\n=== TESTING PARTIAL RECEIPT REIMBURSEMENT ===")

def calculate_with_receipt_rate(days, miles, receipts, receipt_rate=0.5):
    """Test formula with different receipt reimbursement rate"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_component = receipts * receipt_rate
    
    return per_diem + base_miles + max(extra_miles, receipt_component)

receipt_rates = [0.3, 0.4, 0.5, 0.6, 0.7]

for rate in receipt_rates:
    errors = []
    for case_id in worst_case_ids:
        case = cases[case_id]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_with_receipt_rate(days, miles, receipts, rate)
        error = abs(predicted - expected)
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    print(f"Receipt rate {rate}: Average error on worst cases: ${avg_error:.2f}")

# Test our three original problem cases with different receipt rates
print(f"\n=== TESTING RECEIPT RATES ON ORIGINAL PROBLEM CASES ===")
problem_cases = [148, 668, 512]

for rate in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"\nReceipt rate {rate}:")
    for case_id in problem_cases:
        case = cases[case_id]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_with_receipt_rate(days, miles, receipts, rate)
        error = abs(predicted - expected)
        
        print(f"  Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Maybe the issue is that our original max approach doesn't work for all cases
# Let's test a simpler approach: what if it's just additive with caps?
print(f"\n=== TESTING SIMPLE ADDITIVE WITH CAPS ===")

def calculate_simple_additive(days, miles, receipts):
    """Simple additive: per_diem + capped_miles + capped_receipts"""
    per_diem = days * 121.50
    
    # Miles component with different rates for different ranges
    if miles <= 500:
        miles_component = miles * 0.50
    else:
        miles_component = 500 * 0.50 + (miles - 500) * 0.675
    
    # Receipt component - maybe it's capped at per-diem amount?
    receipt_cap = per_diem  # Cap receipts at per-diem amount
    receipts_component = min(receipts, receipt_cap)
    
    return per_diem + miles_component + receipts_component

print("Testing simple additive with receipt cap = per_diem:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_simple_additive(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Test on some of the worst cases too
print(f"\nTesting on some worst cases:")
for case_id in worst_case_ids[:5]:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_simple_additive(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")