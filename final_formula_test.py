#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

def calculate_reimbursement_final(days, miles, receipts):
    """Final formula based on analysis"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipts_component = receipts * 1.0
    
    return per_diem + base_miles + max(extra_miles, receipts_component)

print("=== TESTING FINAL FORMULA ON ALL CASE TYPES ===")

# Test on all cases to get overall performance
all_errors = []
good_cases = 0
total_cases = len(cases)

# Group cases by characteristics for detailed analysis
low_mile_cases = []      # < 500 miles
med_mile_cases = []      # 500-1000 miles  
high_mile_cases = []     # 1000+ miles

for i, case in enumerate(cases):
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = abs(predicted - expected)
    all_errors.append(error)
    
    if error < 50:  # Consider < $50 error as "good"
        good_cases += 1
    
    # Categorize by miles
    if miles < 500:
        low_mile_cases.append((i, error))
    elif miles < 1000:
        med_mile_cases.append((i, error))
    else:
        high_mile_cases.append((i, error))

# Overall statistics
avg_error = sum(all_errors) / len(all_errors)
max_error = max(all_errors)
good_percentage = (good_cases / total_cases) * 100

print(f"OVERALL PERFORMANCE:")
print(f"  Total cases: {total_cases}")
print(f"  Average error: ${avg_error:.2f}")
print(f"  Maximum error: ${max_error:.2f}")
print(f"  Cases with error < $50: {good_cases}/{total_cases} ({good_percentage:.1f}%)")

# Performance by mile category
categories = [
    ("Low miles (<500)", low_mile_cases),
    ("Medium miles (500-1000)", med_mile_cases), 
    ("High miles (1000+)", high_mile_cases)
]

for name, case_list in categories:
    if case_list:
        errors = [error for _, error in case_list]
        avg_err = sum(errors) / len(errors)
        max_err = max(errors)
        good_count = sum(1 for _, error in case_list if error < 50)
        
        print(f"\n{name}: {len(case_list)} cases")
        print(f"  Average error: ${avg_err:.2f}")
        print(f"  Maximum error: ${max_err:.2f}")
        print(f"  Good cases (<$50 error): {good_count}/{len(case_list)} ({100*good_count/len(case_list):.1f}%)")

# Show some examples of good and bad cases
print(f"\n=== EXAMPLES OF BEST CASES ===")
best_cases = sorted(enumerate(all_errors), key=lambda x: x[1])[:10]
for case_idx, error in best_cases:
    case = cases[case_idx]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    predicted = calculate_reimbursement_final(days, miles, receipts)
    
    print(f"Case {case_idx}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

print(f"\n=== EXAMPLES OF WORST CASES ===")
worst_cases = sorted(enumerate(all_errors), key=lambda x: x[1], reverse=True)[:10]
for case_idx, error in worst_cases:
    case = cases[case_idx]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    predicted = calculate_reimbursement_final(days, miles, receipts)
    
    print(f"Case {case_idx}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

# Test specifically on our original problem cases
print(f"\n=== ORIGINAL PROBLEM CASES ===")
problem_cases = [148, 668, 512]
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    predicted = calculate_reimbursement_final(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

# Show the formula breakdown for a few cases
print(f"\n=== FORMULA BREAKDOWN EXAMPLES ===")
example_cases = [148, 25, 500]  # High mile, low mile, medium mile
for case_id in example_cases:
    if case_id < len(cases):
        case = cases[case_id]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        per_diem = days * 121.50
        base_miles = min(miles, 500) * 0.50
        extra_miles = max(0, miles - 500) * 0.675
        receipts_component = receipts * 1.0
        max_component = max(extra_miles, receipts_component)
        predicted = per_diem + base_miles + max_component
        
        print(f"\nCase {case_id}: {days}d, {miles}mi, ${receipts:.2f}")
        print(f"  Per diem: {days} × $121.50 = ${per_diem:.2f}")
        print(f"  Base miles: min({miles}, 500) × $0.50 = ${base_miles:.2f}")
        print(f"  Extra miles: max(0, {miles}-500) × $0.675 = ${extra_miles:.2f}")
        print(f"  Receipts: ${receipts:.2f} × 1.0 = ${receipts_component:.2f}")
        print(f"  Max(extra_miles, receipts): max(${extra_miles:.2f}, ${receipts_component:.2f}) = ${max_component:.2f}")
        print(f"  Total: ${per_diem:.2f} + ${base_miles:.2f} + ${max_component:.2f} = ${predicted:.2f}")
        print(f"  Expected: ${expected:.2f}, Error: ${abs(predicted - expected):.2f}")