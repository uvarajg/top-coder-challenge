#!/usr/bin/env python3
import json

# Load cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== FINAL SOLUTION: RECEIPT CAP BASED ON PER-DIEM ===")

def calculate_reimbursement_final(days, miles, receipts):
    """Final formula with per-diem based receipt cap"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    
    # Key insight: Cap receipts at 1.4x per-diem
    receipt_cap = per_diem * 1.4
    capped_receipts = min(receipts, receipt_cap)
    
    return per_diem + base_miles + max(extra_miles, capped_receipts)

# Test on our problem cases first
problem_cases = [148, 668, 512]
print("Testing final formula on original problem cases:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = abs(predicted - expected)
    
    per_diem = days * 121.50
    receipt_cap = per_diem * 1.4
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")
    print(f"  Per-diem: ${per_diem:.2f}, Receipt cap: ${receipt_cap:.2f}, Actual receipts: ${receipts:.2f}")

# Test on the worst cases
worst_cases = [585, 241, 150, 151, 493]
print(f"\nTesting on previously worst cases:")
for case_id in worst_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Test different cap multipliers to find the best one
print(f"\n=== TESTING DIFFERENT CAP MULTIPLIERS ===")
cap_multipliers = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

for multiplier in cap_multipliers:
    errors = []
    
    # Test on a sample of cases
    test_cases = problem_cases + worst_cases[:3]
    
    for case_id in test_cases:
        case = cases[case_id]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        per_diem = days * 121.50
        base_miles = min(miles, 500) * 0.50
        extra_miles = max(0, miles - 500) * 0.675
        receipt_cap = per_diem * multiplier
        capped_receipts = min(receipts, receipt_cap)
        predicted = per_diem + base_miles + max(extra_miles, capped_receipts)
        
        error = abs(predicted - expected)
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    print(f"Cap multiplier {multiplier}: Average error ${avg_error:.2f}")

# Test overall performance with the best multiplier
print(f"\n=== OVERALL PERFORMANCE TEST ===")
best_multiplier = 1.4  # Based on the results above

all_errors = []
good_cases = 0

for i, case in enumerate(cases):
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_cap = per_diem * best_multiplier
    capped_receipts = min(receipts, receipt_cap)
    predicted = per_diem + base_miles + max(extra_miles, capped_receipts)
    
    error = abs(predicted - expected)
    all_errors.append(error)
    
    if error < 50:
        good_cases += 1

avg_error = sum(all_errors) / len(all_errors)
max_error = max(all_errors)
good_percentage = (good_cases / len(cases)) * 100

print(f"Overall performance with cap multiplier {best_multiplier}:")
print(f"  Average error: ${avg_error:.2f}")
print(f"  Maximum error: ${max_error:.2f}")
print(f"  Cases with error < $50: {good_cases}/{len(cases)} ({good_percentage:.1f}%)")

# Show the final formula clearly
print(f"\n=== FINAL FORMULA ===")
print("REIMBURSEMENT = PER_DIEM + BASE_MILES + MAX(EXTRA_MILES, CAPPED_RECEIPTS)")
print("Where:")
print("  PER_DIEM = days × $121.50")
print("  BASE_MILES = min(miles, 500) × $0.50")
print("  EXTRA_MILES = max(0, miles - 500) × $0.675")
print(f"  CAPPED_RECEIPTS = min(receipts, PER_DIEM × {best_multiplier})")

# Test the formula on our three original high-error cases one more time
print(f"\n=== FINAL VERIFICATION ON ORIGINAL PROBLEM CASES ===")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = abs(predicted - expected)
    error_percentage = (error / expected) * 100
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f}")
    print(f"  Expected: ${expected:.2f}")
    print(f"  Predicted: ${predicted:.2f}")
    print(f"  Error: ${error:.2f} ({error_percentage:.2f}%)")
    print(f"  Status: {'✓ EXCELLENT' if error < 50 else '✓ GOOD' if error < 100 else '✗ NEEDS WORK'}")