#!/usr/bin/env python3

import json
import math

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== ANALYZING FOR CAPS AND THRESHOLDS ===\n")

# Let's look for evidence of caps by examining receipt reimbursement patterns
receipt_analysis = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    
    receipt_analysis.append({
        'days': inp['trip_duration_days'],
        'miles': inp['miles_traveled'],
        'receipts': inp['total_receipts_amount'],
        'expected': exp,
        'receipt_ratio': exp / inp['total_receipts_amount'] if inp['total_receipts_amount'] > 0 else 0
    })

# Sort by receipt amount to look for caps
receipt_analysis.sort(key=lambda x: x['receipts'])

print("=== RECEIPT AMOUNT vs REIMBURSEMENT PATTERNS ===")
print("Looking for evidence of caps...")

# Group by receipt ranges
ranges = [
    (0, 100, "Low receipts (0-100)"),
    (100, 500, "Medium receipts (100-500)"),
    (500, 1000, "High receipts (500-1000)"),
    (1000, 1500, "Very high receipts (1000-1500)"),
    (1500, 2000, "Extremely high receipts (1500-2000)"),
    (2000, 3000, "Ultra high receipts (2000+)")
]

for min_r, max_r, label in ranges:
    in_range = [x for x in receipt_analysis if min_r <= x['receipts'] < max_r]
    if in_range:
        ratios = [x['receipt_ratio'] for x in in_range]
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"{label}: {len(in_range)} cases")
        print(f"  Avg receipt ratio: {avg_ratio:.4f}")
        print(f"  Min receipt ratio: {min_ratio:.4f}")
        print(f"  Max receipt ratio: {max_ratio:.4f}")
        print()

print("=== TESTING POTENTIAL CAPS ===")

# Let's test a hypothesis: maybe there's a cap on receipt reimbursement
# Let's look at cases where receipts are very high and see if there's a pattern

high_receipt_cases = [x for x in receipt_analysis if x['receipts'] > 1500]
print(f"Cases with receipts > $1500: {len(high_receipt_cases)}")

# Check if there might be a percentage cap on receipts
for case in high_receipt_cases[:10]:
    # Try different percentage caps
    for cap_pct in [0.3, 0.4, 0.5, 0.6]:
        capped_receipt_contribution = min(case['receipts'] * cap_pct, case['receipts'])
        print(f"Receipts: ${case['receipts']:.2f}, Expected: ${case['expected']:.2f}")
        print(f"  If {cap_pct*100}% cap: ${capped_receipt_contribution:.2f}")
    print()

print("=== ALTERNATIVE HYPOTHESIS: MINIMUM FUNCTION ===")
print("Maybe reimbursement = min(base_amount, receipt_percentage * receipts)")

# Let's see if there's a pattern like: reimbursement = min(days*X + miles*Y, receipts*Z)
base_rates = []
for case in receipt_analysis:
    # Calculate what the base rate would be if we ignore receipts completely
    # Assume: base = days * day_rate + miles * mile_rate
    # We need to find day_rate and mile_rate
    
    # For now, let's just look at the relationship
    days_miles_combo = case['days'] * 100 + case['miles'] * 0.5  # rough estimate
    receipt_contribution = case['receipts'] * 0.4  # rough estimate
    
    theoretical_min = min(days_miles_combo, receipt_contribution)
    actual = case['expected']
    
    if case['receipts'] > 1000:  # Focus on high receipt cases
        print(f"{case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} receipts -> ${actual:.2f}")
        print(f"  Days+Miles estimate: ${days_miles_combo:.2f}")
        print(f"  Receipt estimate (40%): ${receipt_contribution:.2f}")
        print(f"  Min of both: ${theoretical_min:.2f}")
        print(f"  Actual: ${actual:.2f}")
        print(f"  Ratio actual/min: {actual/theoretical_min:.4f}")
        print()

print("=== LOOKING FOR STEP FUNCTIONS ===")
# Maybe there are discrete steps based on receipt amounts

# Group by similar receipt amounts and see if there are jumps
receipt_groups = {}
for case in receipt_analysis:
    # Round receipts to nearest 100
    rounded = round(case['receipts'] / 100) * 100
    if rounded not in receipt_groups:
        receipt_groups[rounded] = []
    receipt_groups[rounded].append(case)

# Look at groups with multiple cases to see variance
print("Receipt amount groups with multiple cases:")
for amount in sorted(receipt_groups.keys()):
    if len(receipt_groups[amount]) > 1:
        cases = receipt_groups[amount]
        ratios = [c['receipt_ratio'] for c in cases]
        if max(ratios) - min(ratios) < 0.1:  # Low variance suggests a pattern
            print(f"~${amount}: {len(cases)} cases, ratio range: {min(ratios):.4f} - {max(ratios):.4f}")