#!/usr/bin/env python3

import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING HIGH-ERROR CASES ===\n")

# Target cases mentioned by user
target_cases = [
    {"days": 1, "miles": 1082, "receipts": 1809.49, "expected": 446.94},
    {"days": 4, "miles": 69, "receipts": 2321.49, "expected": 322.0},
    {"days": 8, "miles": 795, "receipts": 1645.99, "expected": 644.69},
    {"days": 5, "miles": 516, "receipts": 1878.49, "expected": 669.85}
]

print("Target high-error cases:")
for i, target in enumerate(target_cases):
    print(f"Case {i+1}: {target['days']} days, {target['miles']} miles, ${target['receipts']} receipts -> ${target['expected']}")
    
    # Calculate what coefficients would work for this case alone
    # If formula is: reimbursement = a*days + b*miles + c*receipts
    # We need: target['expected'] = a*target['days'] + b*target['miles'] + c*target['receipts']
    
    # Let's see the ratio of expected to receipts
    receipt_ratio = target['expected'] / target['receipts']
    print(f"  Receipt ratio (expected/receipts): {receipt_ratio:.4f}")
    
    # Let's see reimbursement per day and per mile
    per_day = target['expected'] / target['days']
    per_mile = target['expected'] / target['miles'] if target['miles'] > 0 else 0
    print(f"  Per day: ${per_day:.2f}, Per mile: ${per_mile:.4f}")
    print()

print("\n=== ANALYZING LOW EXPECTED OUTPUT CASES ===\n")

# Find cases with very low expected outputs
low_cases = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    if exp < 200:  # Very low reimbursements
        low_cases.append({
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'], 
            'receipts': inp['total_receipts_amount'],
            'expected': exp
        })

# Sort by expected output
low_cases.sort(key=lambda x: x['expected'])

print("10 lowest expected outputs:")
for i, case in enumerate(low_cases[:10]):
    print(f"Case {i+1}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} receipts -> ${case['expected']}")
    receipt_ratio = case['expected'] / case['receipts'] if case['receipts'] > 0 else 0
    per_day = case['expected'] / case['days']
    per_mile = case['expected'] / case['miles'] if case['miles'] > 0 else 0
    print(f"  Receipt ratio: {receipt_ratio:.4f}, Per day: ${per_day:.2f}, Per mile: ${per_mile:.4f}")
    print()

print("\n=== ANALYZING HIGH RECEIPT CASES ===\n")

# Find cases with very high receipts
high_receipt_cases = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    if inp['total_receipts_amount'] > 2000:  # Very high receipts
        high_receipt_cases.append({
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'], 
            'receipts': inp['total_receipts_amount'],
            'expected': exp
        })

# Sort by receipts amount
high_receipt_cases.sort(key=lambda x: x['receipts'], reverse=True)

print("10 highest receipt cases:")
for i, case in enumerate(high_receipt_cases[:10]):
    print(f"Case {i+1}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} receipts -> ${case['expected']}")
    receipt_ratio = case['expected'] / case['receipts'] if case['receipts'] > 0 else 0
    per_day = case['expected'] / case['days']
    per_mile = case['expected'] / case['miles'] if case['miles'] > 0 else 0
    print(f"  Receipt ratio: {receipt_ratio:.4f}, Per day: ${per_day:.2f}, Per mile: ${per_mile:.4f}")
    print()

print("\n=== PATTERN ANALYSIS ===\n")

# Let's look for potential caps or thresholds
receipt_ratios = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    if inp['total_receipts_amount'] > 0:
        ratio = exp / inp['total_receipts_amount']
        receipt_ratios.append({
            'ratio': ratio,
            'receipts': inp['total_receipts_amount'],
            'expected': exp,
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled']
        })

receipt_ratios.sort(key=lambda x: x['ratio'])

print("Lowest receipt ratios (expected/receipts):")
for i, case in enumerate(receipt_ratios[:10]):
    print(f"Ratio {case['ratio']:.4f}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} -> ${case['expected']:.2f}")

print("\nHighest receipt ratios (expected/receipts):")
for i, case in enumerate(receipt_ratios[-10:]):
    print(f"Ratio {case['ratio']:.4f}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} -> ${case['expected']:.2f}")

print(f"\nMedian receipt ratio: {receipt_ratios[len(receipt_ratios)//2]['ratio']:.4f}")