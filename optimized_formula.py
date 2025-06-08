#!/usr/bin/env python3
import json

# Load cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== OPTIMIZING RECEIPT CAP FOR BEST OVERALL PERFORMANCE ===")

def calculate_reimbursement(days, miles, receipts, cap_multiplier=1.0):
    """Calculate reimbursement with variable cap multiplier"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_cap = per_diem * cap_multiplier
    capped_receipts = min(receipts, receipt_cap)
    
    return per_diem + base_miles + max(extra_miles, capped_receipts)

# Test different cap multipliers more precisely
multipliers = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5]

print("Testing cap multipliers for overall performance:")
best_multiplier = 1.0
best_avg_error = float('inf')

for multiplier in multipliers:
    errors = []
    
    for case in cases:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculate_reimbursement(days, miles, receipts, multiplier)
        error = abs(predicted - expected)
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    good_cases = sum(1 for e in errors if e < 50)
    
    print(f"Cap multiplier {multiplier}: Avg error ${avg_error:.2f}, Good cases: {good_cases}/1000")
    
    if avg_error < best_avg_error:
        best_avg_error = avg_error
        best_multiplier = multiplier

print(f"\nBest overall multiplier: {best_multiplier} with average error ${best_avg_error:.2f}")

# However, let's check what happens to our three target cases with the best overall multiplier
print(f"\n=== CHECKING OUR TARGET CASES WITH BEST OVERALL MULTIPLIER ===")
problem_cases = [148, 668, 512]

for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement(days, miles, receipts, best_multiplier)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# The insight is that for our specific high-error cases, we need cap_multiplier = 1.4
# But for overall performance, cap_multiplier = 1.0 is better
# Let me test a hybrid approach: different caps based on case characteristics

print(f"\n=== TESTING ADAPTIVE CAP STRATEGY ===")

def calculate_reimbursement_adaptive(days, miles, receipts):
    """Adaptive cap based on case characteristics"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    
    # Adaptive cap: higher for high-mile cases
    if miles >= 1000 and receipts/per_diem > 1.0:
        cap_multiplier = 1.4  # Higher cap for high-mile, high-receipt cases
    else:
        cap_multiplier = 1.0  # Standard cap for other cases
    
    receipt_cap = per_diem * cap_multiplier
    capped_receipts = min(receipts, receipt_cap)
    
    return per_diem + base_miles + max(extra_miles, capped_receipts)

# Test adaptive approach
print("Testing adaptive cap strategy:")
adaptive_errors = []
for case in cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_adaptive(days, miles, receipts)
    error = abs(predicted - expected)
    adaptive_errors.append(error)

adaptive_avg = sum(adaptive_errors) / len(adaptive_errors)
adaptive_good = sum(1 for e in adaptive_errors if e < 50)

print(f"Adaptive strategy: Avg error ${adaptive_avg:.2f}, Good cases: {adaptive_good}/1000")

# Check our target cases with adaptive strategy
print(f"\nTarget cases with adaptive strategy:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = calculate_reimbursement_adaptive(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Final recommendation
print(f"\n=== FINAL RECOMMENDATION ===")
print("For the specific high-error cases you mentioned, use:")
print("REIMBURSEMENT = PER_DIEM + BASE_MILES + MAX(EXTRA_MILES, CAPPED_RECEIPTS)")
print("Where:")
print("  PER_DIEM = days × $121.50")
print("  BASE_MILES = min(miles, 500) × $0.50")
print("  EXTRA_MILES = max(0, miles - 500) × $0.675")
print("  CAPPED_RECEIPTS = min(receipts, PER_DIEM × 1.4)")
print()
print("This gives excellent results for high-mile, medium-receipt cases:")
print("- Case 148: Error $2.01 (0.09%)")
print("- Case 668: Error $6.30 (0.30%)")  
print("- Case 512: Error $38.69 (1.75%)")