#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

# Focus on the specific high-error cases mentioned
target_cases = [
    {"case": 148, "days": 7, "miles": 1006, "receipts": 1181.33, "expected": 2279.82},
    {"case": 668, "days": 7, "miles": 1033, "receipts": 1013.03, "expected": 2119.83},
    {"case": 512, "days": 8, "miles": 1025, "receipts": 1031.33, "expected": 2214.64}
]

print("=== ANALYZING SPECIFIC HIGH-ERROR CASES ===")
for target in target_cases:
    case = cases[target["case"]]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    print(f"\nCase {target['case']}: {days} days, {miles} miles, ${receipts:.2f} receipts -> Expected: ${expected:.2f}")
    
    # Test different formulas
    per_diem = days * 121.50
    
    # Test if it's additive: per_diem + miles*rate + receipts*multiplier
    remaining_after_perdiem = expected - per_diem
    
    # Test various mile rates
    for mile_rate in [0.25, 0.50, 0.675, 1.0, 1.25, 1.50]:
        miles_component = miles * mile_rate
        remaining_after_miles = remaining_after_perdiem - miles_component
        receipts_multiplier = remaining_after_miles / receipts if receipts > 0 else 0
        
        total_calc = per_diem + miles_component + (receipts * receipts_multiplier)
        
        print(f"  Mile rate ${mile_rate:.3f}: Perdiem=${per_diem:.2f} + Miles=${miles_component:.2f} + Receipts*{receipts_multiplier:.3f}=${remaining_after_miles:.2f} = ${total_calc:.2f}")

# Now look for patterns in high-mile cases
print("\n=== CHECKING FOR MILE THRESHOLD PATTERNS ===")

# Group cases by mile ranges and look for consistent patterns
high_mile_cases = []
for i, case in enumerate(cases):
    miles = case['input']['miles_traveled']
    if miles >= 1000:
        days = case['input']['trip_duration_days']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        high_mile_cases.append({
            'id': i,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected
        })

# Test if there's a consistent additive formula
print(f"\nTesting additive formula on {len(high_mile_cases)} high-mile cases...")

# Test different combinations
test_formulas = [
    {"per_diem": 121.50, "mile_rate": 0.50, "receipt_rate": 1.0},
    {"per_diem": 121.50, "mile_rate": 0.675, "receipt_rate": 1.0},
    {"per_diem": 121.50, "mile_rate": 1.0, "receipt_rate": 1.0},
    {"per_diem": 121.50, "mile_rate": 1.0, "receipt_rate": 0.5},
    {"per_diem": 121.50, "mile_rate": 1.25, "receipt_rate": 0.5},
]

for formula in test_formulas:
    errors = []
    for case in high_mile_cases[:20]:  # Test on first 20 cases
        predicted = (case['days'] * formula['per_diem'] + 
                    case['miles'] * formula['mile_rate'] + 
                    case['receipts'] * formula['receipt_rate'])
        error = abs(predicted - case['expected'])
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    
    print(f"Formula: ${formula['per_diem']:.2f}/day + ${formula['mile_rate']:.3f}/mile + ${formula['receipt_rate']:.1f}*receipts")
    print(f"  Avg error: ${avg_error:.2f}, Max error: ${max_error:.2f}")

# Check if there might be a multiplicative component
print("\n=== TESTING MULTIPLICATIVE EFFECTS ===")
for case in target_cases:
    case_data = cases[case["case"]]
    days = case_data['input']['trip_duration_days']
    miles = case_data['input']['miles_traveled']
    receipts = case_data['input']['total_receipts_amount']
    expected = case_data['expected_output']
    
    print(f"\nCase {case['case']}: {days}d, {miles}mi, ${receipts:.2f} -> ${expected:.2f}")
    
    # Test: base + days*rate + miles*rate + receipts*rate + interaction_term
    base_calc = days * 121.50 + miles * 0.675 + receipts * 1.0
    
    # What if there's a bonus for high miles with high receipts?
    if miles > 1000 and receipts > 1000:
        # Test various interaction effects
        mile_receipt_product = miles * receipts / 1000000  # Scale down the product
        print(f"  Base calculation: ${base_calc:.2f}")
        print(f"  Miles*Receipts/1M: {mile_receipt_product:.2f}")
        print(f"  Difference needed: ${expected - base_calc:.2f}")
        
        # Test if there's a miles threshold bonus
        miles_over_1000 = miles - 1000
        potential_bonus = miles_over_1000 * 1.0  # $1 per mile over 1000?
        print(f"  Miles over 1000: {miles_over_1000}, potential $1/mile bonus: ${potential_bonus:.2f}")