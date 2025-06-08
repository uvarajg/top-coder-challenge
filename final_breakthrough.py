#!/usr/bin/env python3
import json

# Load cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING RECEIPT PENALTY HYPOTHESIS ===")

# Based on the analysis, I think the formula might be:
# per_diem + base_miles + max(extra_miles, receipts) - receipt_penalty

def test_receipt_penalty_formula(days, miles, receipts):
    """Test formula with receipt penalty for very high receipts"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_component = receipts * 1.0
    
    # If receipts are very high (more than per_diem), apply penalty?
    if receipts > per_diem:
        penalty = (receipts - per_diem) * 0.5  # 50% penalty on excess receipts
        return per_diem + base_miles + max(extra_miles, receipt_component) - penalty
    else:
        return per_diem + base_miles + max(extra_miles, receipt_component)

# Test on our known cases
problem_cases = [148, 668, 512]
print("Testing receipt penalty on original problem cases:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = test_receipt_penalty_formula(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# But wait - let me try a completely different approach
# What if it's not max at all, but a weighted combination based on the ratio?

print(f"\n=== TESTING WEIGHTED COMBINATION HYPOTHESIS ===")

def test_weighted_combination(days, miles, receipts):
    """Test weighted combination of miles and receipts"""
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_component = receipts * 1.0
    
    # Weight based on which is larger
    if extra_miles > receipt_component:
        # Miles dominant: use more miles, less receipts
        weighted_component = extra_miles * 0.8 + receipt_component * 0.2
    else:
        # Receipts dominant: use more receipts, less miles  
        weighted_component = extra_miles * 0.2 + receipt_component * 0.8
    
    return per_diem + base_miles + weighted_component

print("Testing weighted combination:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = test_weighted_combination(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Let me try one more approach: what if there's a completely different formula for high-receipt cases?
print(f"\n=== TESTING DIFFERENT FORMULA FOR HIGH RECEIPTS ===")

def test_high_receipt_formula(days, miles, receipts):
    """Different handling for high receipt cases"""
    per_diem = days * 121.50
    
    # Standard miles calculation
    if miles <= 500:
        miles_component = miles * 0.50
    else:
        miles_component = 500 * 0.50 + (miles - 500) * 0.675
    
    # If receipts are more than 2x per_diem, use different formula
    if receipts > 2 * per_diem:
        # High receipts: maybe use fixed rate regardless of amount?
        receipt_component = per_diem * 0.5  # Fixed 50% of per_diem
    elif receipts > per_diem:
        # Medium receipts: use receipts but with penalty
        receipt_component = per_diem + (receipts - per_diem) * 0.2
    else:
        # Low receipts: use full amount
        receipt_component = receipts
    
    return per_diem + miles_component + receipt_component

print("Testing high receipt formula:")
for case_id in problem_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    predicted = test_high_receipt_formula(days, miles, receipts)
    error = abs(predicted - expected)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")

# Let me go back to the basic insight: the current max formula works perfectly for our 3 cases
# Maybe I need to understand WHY it works for them but not others
print(f"\n=== ANALYZING WHY MAX FORMULA WORKS FOR SPECIFIC CASES ===")

def analyze_max_formula_components(case_id):
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem = days * 121.50
    base_miles = min(miles, 500) * 0.50
    extra_miles = max(0, miles - 500) * 0.675
    receipt_component = receipts * 1.0
    
    chosen = "receipts" if receipt_component > extra_miles else "extra_miles"
    max_component = max(extra_miles, receipt_component)
    predicted = per_diem + base_miles + max_component
    
    print(f"\nCase {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> Expected: ${expected:.2f}")
    print(f"  Per diem: ${per_diem:.2f}")
    print(f"  Base miles: ${base_miles:.2f}")
    print(f"  Extra miles: ${extra_miles:.2f}")
    print(f"  Receipts: ${receipt_component:.2f}")
    print(f"  Max chose: {chosen} (${max_component:.2f})")
    print(f"  Predicted: ${predicted:.2f}")
    print(f"  Error: ${abs(predicted - expected):.2f}")
    
    # Calculate ratio of receipts to per_diem
    receipt_ratio = receipts / per_diem if per_diem > 0 else 0
    print(f"  Receipt/Per-diem ratio: {receipt_ratio:.2f}")

# Analyze our good cases
for case_id in problem_cases:
    analyze_max_formula_components(case_id)

# Compare with some bad cases
worst_cases = [585, 241, 150]
print(f"\n=== COMPARING WITH WORST CASES ===")
for case_id in worst_cases:
    analyze_max_formula_components(case_id)