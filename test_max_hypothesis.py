#!/usr/bin/env python3

import json
import math

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== TESTING MAXIMUM/MINIMUM FUNCTION HYPOTHESIS ===\n")

# Maybe the formula involves max() or min() operations
# Let's test: reimbursement = max(base_amount, min(receipt_amount, cap))

# Try to find base rates from very low receipt cases
print("Finding base rates from minimal receipt cases...")

minimal_receipt_cases = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    if inp['total_receipts_amount'] < 25:  # Very minimal receipts
        minimal_receipt_cases.append({
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'expected': exp
        })

print(f"Found {len(minimal_receipt_cases)} cases with receipts < $25")

# Try to estimate day_rate and mile_rate
# Using simple linear regression approach
total_days = sum(c['days'] for c in minimal_receipt_cases)
total_miles = sum(c['miles'] for c in minimal_receipt_cases)
total_expected = sum(c['expected'] for c in minimal_receipt_cases)

print(f"Total days: {total_days}, total miles: {total_miles}, total expected: ${total_expected:.2f}")

# Rough estimate: assume day_rate = 100, solve for mile_rate
estimated_day_contribution = total_days * 100
remaining = total_expected - estimated_day_contribution
estimated_mile_rate = remaining / total_miles if total_miles > 0 else 0

print(f"If day_rate = 100, then mile_rate ≈ {estimated_mile_rate:.4f}")

# Now test various hypotheses on the problematic cases
target_cases = [
    {"days": 1, "miles": 1082, "receipts": 1809.49, "expected": 446.94},
    {"days": 4, "miles": 69, "receipts": 2321.49, "expected": 322.0},
    {"days": 8, "miles": 795, "receipts": 1645.99, "expected": 644.69},
    {"days": 5, "miles": 516, "receipts": 1878.49, "expected": 669.85}
]

print(f"\nTesting hypotheses with day_rate=100, mile_rate={estimated_mile_rate:.4f}")

hypotheses = [
    ("Linear", lambda d, m, r: 100*d + estimated_mile_rate*m + 0.4*r),
    ("Cap at 800", lambda d, m, r: 100*d + estimated_mile_rate*m + min(800, 0.4*r)),
    ("Cap at 600", lambda d, m, r: 100*d + estimated_mile_rate*m + min(600, 0.4*r)),
    ("Cap at 400", lambda d, m, r: 100*d + estimated_mile_rate*m + min(400, 0.4*r)),
    ("Square root", lambda d, m, r: 100*d + estimated_mile_rate*m + 100*math.sqrt(r)),
    ("Logarithmic", lambda d, m, r: 100*d + estimated_mile_rate*m + 200*math.log(r+1)),
    ("Max function", lambda d, m, r: max(100*d + estimated_mile_rate*m, min(1200, 0.4*r))),
    ("Min function", lambda d, m, r: min(100*d + estimated_mile_rate*m + 1000, 0.4*r + 200)),
    ("Two-tier receipts", lambda d, m, r: 100*d + estimated_mile_rate*m + (0.6*min(r, 1000) + 0.2*max(0, r-1000))),
]

print("\nTesting different formulas:")
for name, formula in hypotheses:
    print(f"\n{name}:")
    total_error = 0
    for i, case in enumerate(target_cases):
        predicted = formula(case['days'], case['miles'], case['receipts'])
        error = abs(predicted - case['expected'])
        total_error += error
        print(f"  Case {i+1}: Predicted ${predicted:.2f}, Actual ${case['expected']:.2f}, Error ${error:.2f}")
    print(f"  Total error: ${total_error:.2f}")

print("\n=== DETAILED ANALYSIS OF RECEIPT PATTERNS ===")

# Look for evidence of step functions or tiers
receipt_bins = {}
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    
    # Bin by receipt amount (rounded to nearest 500)
    bin_size = 500
    receipt_bin = round(inp['total_receipts_amount'] / bin_size) * bin_size
    
    if receipt_bin not in receipt_bins:
        receipt_bins[receipt_bin] = []
    
    receipt_bins[receipt_bin].append({
        'days': inp['trip_duration_days'],
        'miles': inp['miles_traveled'],
        'receipts': inp['total_receipts_amount'],
        'expected': exp,
        'ratio': exp / inp['total_receipts_amount'] if inp['total_receipts_amount'] > 0 else 0
    })

print("Receipt reimbursement by amount bins:")
for bin_amount in sorted(receipt_bins.keys()):
    cases_in_bin = receipt_bins[bin_amount]
    if len(cases_in_bin) >= 5:  # Only show bins with enough data
        ratios = [c['ratio'] for c in cases_in_bin]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"${bin_amount:.0f}: {len(cases_in_bin)} cases, avg ratio {avg_ratio:.4f}, range {min_ratio:.4f}-{max_ratio:.4f}")

print("\n=== FINAL INSIGHTS ===")
print("Based on this analysis:")
print("1. Receipt reimbursement definitely has diminishing returns")
print("2. There may be caps or tiers around $1000-1500 receipt amounts") 
print("3. Your problematic cases all have receipts > $1600 with very low ratios")
print("4. Try formulas with caps, square root, or logarithmic receipt functions")
print("5. The base day/mile rates seem reasonable around 100/0.5-0.7")