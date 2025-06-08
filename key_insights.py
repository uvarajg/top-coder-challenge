#!/usr/bin/env python3

import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== KEY INSIGHTS FROM HIGH-ERROR CASES ANALYSIS ===\n")

# The high-error cases you mentioned
target_cases = [
    {"days": 1, "miles": 1082, "receipts": 1809.49, "expected": 446.94},
    {"days": 4, "miles": 69, "receipts": 2321.49, "expected": 322.0},
    {"days": 8, "miles": 795, "receipts": 1645.99, "expected": 644.69},
    {"days": 5, "miles": 516, "receipts": 1878.49, "expected": 669.85}
]

print("PATTERN 1: Receipt reimbursement appears to have diminishing returns")
print("Notice how receipt ratios (expected/receipts) get smaller as receipt amounts increase:")
print()

# Show receipt ratio patterns
receipt_brackets = [
    (0, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)
]

for min_r, max_r in receipt_brackets:
    matching = []
    for case in cases:
        inp = case['input']
        exp = case['expected_output']
        if min_r <= inp['total_receipts_amount'] < max_r:
            ratio = exp / inp['total_receipts_amount'] if inp['total_receipts_amount'] > 0 else 0
            matching.append(ratio)
    
    if matching:
        avg_ratio = sum(matching) / len(matching)
        print(f"Receipts ${min_r}-${max_r}: Average ratio = {avg_ratio:.4f}")

print()
print("PATTERN 2: Your target cases have very low receipt ratios")
for i, case in enumerate(target_cases):
    ratio = case['expected'] / case['receipts']
    print(f"Case {i+1}: Receipt ratio = {ratio:.4f} (very low!)")

print()
print("PATTERN 3: Looking for evidence of caps or maximum reimbursement amounts")

# Let's check if there might be caps based on days/miles
print()
print("Testing hypothesis: Maybe there's a cap like min(base_rate, receipt_percentage)")

# Try to reverse engineer coefficients
print("\nReverse engineering from low-receipt cases (where receipts shouldn't be the limiting factor):")

# Find cases with very low receipts where the formula should be dominated by days/miles
low_receipt_cases = []
for case in cases:
    inp = case['input']
    exp = case['expected_output']
    if inp['total_receipts_amount'] < 50:  # Very low receipts
        low_receipt_cases.append({
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'expected': exp
        })

# Try to solve for day_rate and mile_rate using these cases
print("\nLow receipt cases (where base rate should dominate):")
for case in low_receipt_cases[:10]:
    print(f"{case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} -> ${case['expected']:.2f}")
    # If expected = day_rate * days + mile_rate * miles + receipt_rate * receipts
    # and receipts are tiny, then: expected ≈ day_rate * days + mile_rate * miles
    if case['miles'] > 0:
        # Assume day_rate = 100, solve for mile_rate
        implied_mile_rate = (case['expected'] - 100 * case['days']) / case['miles']
        print(f"  If day_rate=100, then mile_rate≈{implied_mile_rate:.4f}")

print()
print("HYPOTHESIS: Non-linear receipt reimbursement with caps")
print("Maybe the formula is something like:")
print("reimbursement = day_rate * days + mile_rate * miles + min(receipt_cap, receipt_rate * receipts)")
print()

# Test this hypothesis on target cases
print("Testing cap hypothesis on your high-error cases:")
day_rate = 100  # Rough estimate
mile_rate = 0.5  # Rough estimate

for i, case in enumerate(target_cases):
    base_amount = day_rate * case['days'] + mile_rate * case['miles']
    print(f"\nCase {i+1}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f}")
    print(f"  Base amount (days+miles): ${base_amount:.2f}")
    print(f"  Actual expected: ${case['expected']:.2f}")
    print(f"  Difference: ${case['expected'] - base_amount:.2f}")
    
    # What receipt rate would work?
    receipt_contribution = case['expected'] - base_amount
    if receipt_contribution > 0:
        implied_receipt_rate = receipt_contribution / case['receipts']
        print(f"  Implied receipt rate: {implied_receipt_rate:.4f}")
        # Is this a reasonable cap?
        print(f"  This is {implied_receipt_rate*100:.2f}% of receipts")

print()
print("ALTERNATIVE HYPOTHESIS: Tiered or percentage-based receipt reimbursement")
print("Maybe receipts are reimbursed at different rates depending on amount:")
print("- First $X at rate1")
print("- Next $Y at rate2") 
print("- Remainder at rate3")
print()

# Look for evidence of tiered rates
print("Analyzing receipt amounts around your problematic cases:")
problematic_receipts = [1809.49, 2321.49, 1645.99, 1878.49]

for target_receipt in problematic_receipts:
    print(f"\nCases with receipts near ${target_receipt:.2f}:")
    similar_cases = []
    for case in cases:
        inp = case['input']
        exp = case['expected_output']
        if abs(inp['total_receipts_amount'] - target_receipt) < 50:
            similar_cases.append({
                'days': inp['trip_duration_days'],
                'miles': inp['miles_traveled'],
                'receipts': inp['total_receipts_amount'],
                'expected': exp,
                'ratio': exp / inp['total_receipts_amount']
            })
    
    # Sort by receipt amount
    similar_cases.sort(key=lambda x: x['receipts'])
    
    for case in similar_cases[:5]:  # Show first 5
        print(f"  {case['days']}d, {case['miles']}mi, ${case['receipts']:.2f} -> ${case['expected']:.2f} (ratio: {case['ratio']:.4f})")

print()
print("=== RECOMMENDATIONS ===")
print()
print("1. RECEIPT CAPS: Your linear formula fails because receipt reimbursement")
print("   has diminishing returns or caps. High receipt amounts get much lower")
print("   reimbursement rates.")
print()
print("2. TRY THESE FORMULAS:")
print("   a) reimbursement = day_rate * days + mile_rate * miles + min(cap, receipt_rate * receipts)")
print("   b) reimbursement = day_rate * days + mile_rate * miles + f(receipts)")
print("      where f(receipts) is a sublinear function like sqrt() or log()")
print()
print("3. SPECIFIC VALUES TO TEST:")
print("   - day_rate around 80-120")
print("   - mile_rate around 0.4-0.8") 
print("   - receipt_rate around 0.3-0.5 for low amounts")
print("   - receipt cap around $800-1200")
print()
print("4. The pattern suggests that once receipts exceed ~$1500,")
print("   the reimbursement rate drops significantly, possibly to")
print("   protect against excessive claims.")