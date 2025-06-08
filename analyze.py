import json
import numpy as np
from collections import Counter

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Extract data
trip_days = []
miles = []
receipts = []
reimbursements = []

for case in data:
    trip_days.append(case['input']['trip_duration_days'])
    miles.append(case['input']['miles_traveled'])
    receipts.append(case['input']['total_receipts_amount'])
    reimbursements.append(case['expected_output'])

trip_days = np.array(trip_days)
miles = np.array(miles)
receipts = np.array(receipts)
reimbursements = np.array(reimbursements)

print('=== EFFICIENCY-BASED PATTERNS ===')

# Calculate efficiency metrics
miles_per_day = miles / trip_days

for category, (min_mpd, max_mpd) in [('low', (0, 50)), ('medium', (50, 100)), ('high', (100, 1000))]:
    mask = (miles_per_day >= min_mpd) & (miles_per_day < max_mpd)
    if np.sum(mask) > 10:
        cat_reimburs = reimbursements[mask]
        cat_days = trip_days[mask]
        cat_miles = miles[mask]
        
        print(f'\n{category.upper()} efficiency trips ({np.sum(mask)} cases):')
        print(f'  Miles/day range: {miles_per_day[mask].min():.1f} - {miles_per_day[mask].max():.1f}')
        print(f'  Reimbursement/day: {(cat_reimburs/cat_days).mean():.2f} ± {(cat_reimburs/cat_days).std():.2f}')
        print(f'  Reimbursement/mile: {(cat_reimburs/cat_miles).mean():.3f} ± {(cat_reimburs/cat_miles).std():.3f}')

print('\n=== DAILY RATE ANALYSIS ===')

# Look for potential daily caps
daily_rates = reimbursements / trip_days
print(f'Daily rate percentiles:')
for p in [50, 75, 90, 95, 99]:
    print(f'  {p}th percentile: ${np.percentile(daily_rates, p):.2f}')

# Check for common patterns in exact calculations
print('\n=== TESTING EXACT FORMULA PATTERNS ===')

# Test if it's based on a combination with specific rates
for base_per_day in [50, 75, 100, 125, 150]:
    for mileage_rate in [0.50, 0.56, 0.60]:
        for receipt_mult in [0.25, 0.5, 0.75, 1.0]:
            predicted = base_per_day * trip_days + mileage_rate * miles + receipt_mult * receipts
            exact_matches = np.sum(np.abs(predicted - reimbursements) < 0.01)
            if exact_matches > 10:
                print(f'POTENTIAL MATCH: ${base_per_day}/day + ${mileage_rate}/mile + {receipt_mult}*receipts = {exact_matches} exact matches')

print('\n=== LOOKING FOR SPECIAL CASES ===')

# Find cases with very high or low reimbursement ratios
total_naive = trip_days * 100 + miles * 0.56 + receipts
ratio = reimbursements / total_naive

# High ratio cases (potentially have bonuses)
high_ratio_mask = ratio > 1.5
if np.sum(high_ratio_mask) > 0:
    print(f'\nHigh ratio cases ({np.sum(high_ratio_mask)} cases):')
    print(f'  Ratio range: {ratio[high_ratio_mask].min():.2f} - {ratio[high_ratio_mask].max():.2f}')
    print(f'  Trip days: {trip_days[high_ratio_mask].min()} - {trip_days[high_ratio_mask].max()}')
    print(f'  Miles/day: {miles_per_day[high_ratio_mask].min():.1f} - {miles_per_day[high_ratio_mask].max():.1f}')

# Low ratio cases (potentially have caps)
low_ratio_mask = ratio < 0.7
if np.sum(low_ratio_mask) > 0:
    print(f'\nLow ratio cases ({np.sum(low_ratio_mask)} cases):')
    print(f'  Ratio range: {ratio[low_ratio_mask].min():.2f} - {ratio[low_ratio_mask].max():.2f}')
    print(f'  Trip days: {trip_days[low_ratio_mask].min()} - {trip_days[low_ratio_mask].max()}')
    print(f'  Miles/day: {miles_per_day[low_ratio_mask].min():.1f} - {miles_per_day[low_ratio_mask].max():.1f}')