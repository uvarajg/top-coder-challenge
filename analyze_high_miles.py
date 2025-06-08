#!/usr/bin/env python3
import json

# Load the public cases
with open('/workspaces/top-coder-challenge/public_cases.json', 'r') as f:
    cases = json.load(f)

print(f"Total cases: {len(cases)}")

# Find high-mileage cases (1000+ miles)
high_mile_cases = []
for i, case in enumerate(cases):
    miles = case['input']['miles_traveled']
    if miles >= 1000:
        high_mile_cases.append({
            'case_id': i,
            'days': case['input']['trip_duration_days'],
            'miles': miles,
            'receipts': case['input']['total_receipts_amount'],
            'expected': case['expected_output']
        })

print(f"\nFound {len(high_mile_cases)} cases with 1000+ miles:")
for case in high_mile_cases:
    print(f"Case {case['case_id']}: {case['days']} days, {case['miles']} miles, ${case['receipts']:.2f} receipts -> Expected: ${case['expected']:.2f}")

# Calculate effective per-mile rates
print("\n=== Analyzing effective per-mile rates ===")
for case in high_mile_cases:
    # Try different formulas to see what might work
    days_rate = case['days'] * 121.50  # Current per-diem rate
    miles_rate_50 = case['miles'] * 0.50  # $0.50 per mile
    miles_rate_67 = case['miles'] * 0.675  # $0.675 per mile
    receipts_rate = case['receipts'] * 1.0  # 100% receipt reimbursement
    
    # Test additive formula: days + miles + receipts
    additive_50 = days_rate + miles_rate_50 + receipts_rate
    additive_67 = days_rate + miles_rate_67 + receipts_rate
    
    # Calculate what per-mile rate would be needed
    remaining_after_days_receipts = case['expected'] - days_rate - receipts_rate
    needed_mile_rate = remaining_after_days_receipts / case['miles']
    
    print(f"\nCase {case['case_id']}:")
    print(f"  Days component: ${days_rate:.2f}")
    print(f"  Receipts component: ${receipts_rate:.2f}")
    print(f"  Expected: ${case['expected']:.2f}")
    print(f"  Remaining for miles: ${remaining_after_days_receipts:.2f}")
    print(f"  Needed per-mile rate: ${needed_mile_rate:.4f}")
    print(f"  Additive with $0.50/mile: ${additive_50:.2f}")
    print(f"  Additive with $0.675/mile: ${additive_67:.2f}")

# Check if there's a miles threshold pattern
print("\n=== Checking for mile thresholds ===")
mile_ranges = [
    (0, 500),
    (500, 750),
    (750, 1000),
    (1000, 1250),
    (1250, 1500),
    (1500, float('inf'))
]

for min_miles, max_miles in mile_ranges:
    range_cases = []
    for i, case in enumerate(cases):
        miles = case['input']['miles_traveled']
        if min_miles <= miles < max_miles:
            days = case['input']['trip_duration_days']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Calculate effective rate per mile
            days_component = days * 121.50
            remaining = expected - days_component - receipts
            if miles > 0:
                effective_mile_rate = remaining / miles
                range_cases.append(effective_mile_rate)
    
    if range_cases:
        avg_rate = sum(range_cases) / len(range_cases)
        print(f"Miles {min_miles}-{max_miles}: {len(range_cases)} cases, avg rate: ${avg_rate:.4f}/mile")