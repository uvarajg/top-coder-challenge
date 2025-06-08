#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print("=== REVERSE ENGINEERING EXACT COEFFICIENTS ===")

# Focus on the three problem cases and see what exact coefficients work
target_cases = [148, 668, 512]

print("For the three high-error cases, let's see what exact coefficients work:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    print(f"\nCase {case_id}: {days}d, {miles}mi, ${receipts:.2f} -> ${expected:.2f}")
    
    # If formula is: days*D + miles*M + receipts*R = expected
    # We know D = 121.50, so: miles*M + receipts*R = expected - days*121.50
    
    per_diem_component = days * 121.50
    remaining = expected - per_diem_component
    
    print(f"  After per-diem (${per_diem_component:.2f}), remaining: ${remaining:.2f}")
    print(f"  Need: {miles}*M + {receipts:.2f}*R = {remaining:.2f}")
    
    # Try different mile rates and see what receipt rate is needed
    for mile_rate in [0.25, 0.50, 0.675, 1.0, 1.25]:
        miles_component = miles * mile_rate
        receipts_needed = remaining - miles_component
        receipt_rate = receipts_needed / receipts if receipts > 0 else 0
        
        print(f"    If mile_rate=${mile_rate:.3f}: receipt_rate={receipt_rate:.3f}")

# Now let's look at patterns across ALL high-mile cases
print("\n=== ANALYZING ALL HIGH-MILE CASES FOR PATTERNS ===")

high_mile_cases = [(i, case) for i, case in enumerate(cases) if case['input']['miles_traveled'] >= 1000]

# For each case, calculate what the exact coefficients would need to be
coefficients = []
for case_id, case in high_mile_cases:
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    per_diem_component = days * 121.50
    remaining = expected - per_diem_component
    
    # Test with mile_rate = 1.0
    miles_component = miles * 1.0
    receipts_component = remaining - miles_component
    receipt_rate = receipts_component / receipts if receipts > 0 else 0
    
    coefficients.append({
        'case_id': case_id,
        'days': days,
        'miles': miles,
        'receipts': receipts,
        'receipt_rate': receipt_rate,
        'remaining': remaining
    })

# Look for patterns in receipt rates
print(f"\nAnalyzing receipt rates for {len(coefficients)} high-mile cases:")
receipt_rates = [c['receipt_rate'] for c in coefficients if c['receipts'] > 0]
print(f"Receipt rate range: {min(receipt_rates):.3f} to {max(receipt_rates):.3f}")
print(f"Average receipt rate: {sum(receipt_rates)/len(receipt_rates):.3f}")

# Group by receipt amount ranges
low_receipt_cases = [c for c in coefficients if c['receipts'] < 500]
med_receipt_cases = [c for c in coefficients if 500 <= c['receipts'] < 1500]
high_receipt_cases = [c for c in coefficients if c['receipts'] >= 1500]

for name, group in [("Low receipts (<$500)", low_receipt_cases), 
                   ("Med receipts ($500-$1500)", med_receipt_cases),
                   ("High receipts (≥$1500)", high_receipt_cases)]:
    if group:
        rates = [c['receipt_rate'] for c in group if c['receipts'] > 0]
        if rates:
            print(f"{name}: avg rate = {sum(rates)/len(rates):.3f} ({len(rates)} cases)")

# Test another hypothesis: maybe it's not just additive
print("\n=== TESTING NON-ADDITIVE FORMULAS ===")

# What if it's: per_diem + max(miles*rate, receipts*rate)?
def test_max_formula(days, miles, receipts):
    per_diem = days * 121.50
    miles_component = miles * 0.675
    receipts_component = receipts * 1.0
    return per_diem + max(miles_component, receipts_component)

# What if it's: per_diem + miles*rate + min(receipts, cap)*rate?
def test_capped_receipts(days, miles, receipts):
    per_diem = days * 121.50
    miles_component = miles * 0.675
    capped_receipts = min(receipts, 1000)  # Cap receipts at $1000?
    receipts_component = capped_receipts * 1.0
    return per_diem + miles_component + receipts_component

print("Testing alternative formulas on problem cases:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    max_result = test_max_formula(days, miles, receipts)
    capped_result = test_capped_receipts(days, miles, receipts)
    
    print(f"Case {case_id}: Expected ${expected:.2f}")
    print(f"  Max formula: ${max_result:.2f} (error: ${abs(max_result - expected):.2f})")
    print(f"  Capped receipts: ${capped_result:.2f} (error: ${abs(capped_result - expected):.2f})")

# One more test: what if there's a minimum reimbursement per mile?
print("\n=== TESTING MINIMUM MILE REIMBURSEMENT ===")
def test_minimum_mile_rate(days, miles, receipts):
    per_diem = days * 121.50
    
    # Minimum reimbursement: $1.50 per mile for 1000+ mile trips?
    if miles >= 1000:
        mile_component = miles * 1.50
        # But maybe receipts get a lower rate to compensate
        receipt_component = receipts * 0.25
    else:
        mile_component = miles * 0.675
        receipt_component = receipts * 1.0
    
    return per_diem + mile_component + receipt_component

print("Testing minimum mile rate hypothesis:")
for case_id in target_cases:
    case = cases[case_id]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled']
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    min_mile_result = test_minimum_mile_rate(days, miles, receipts)
    
    print(f"Case {case_id}: Expected ${expected:.2f}, Min mile rate: ${min_mile_result:.2f}, Error: ${abs(min_mile_result - expected):.2f}")