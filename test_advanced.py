import json
import numpy as np
from scipy.optimize import minimize

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

print('=== TESTING EFFICIENCY-BASED FORMULA ===')

# Test hypothesis: reimbursement varies based on miles per day efficiency
miles_per_day = miles / trip_days

def efficiency_based_formula(params):
    base_per_day, mileage_rate, receipt_factor, efficiency_bonus, efficiency_threshold = params
    
    predicted = np.zeros(len(reimbursements))
    for i in range(len(reimbursements)):
        mpd = miles_per_day[i]
        base = base_per_day * trip_days[i]
        mileage = mileage_rate * miles[i]
        receipt_comp = receipt_factor * receipts[i]
        
        # Efficiency bonus for high miles per day
        if mpd > efficiency_threshold:
            efficiency_mult = 1 + efficiency_bonus * (mpd - efficiency_threshold) / 100
        else:
            efficiency_mult = 1
            
        predicted[i] = (base + mileage + receipt_comp) * efficiency_mult
    
    return np.sum((predicted - reimbursements) ** 2)

# Optimize parameters
result = minimize(efficiency_based_formula, [100, 0.56, 0.5, 0.01, 100], 
                 bounds=[(10, 200), (0.1, 1.0), (0.1, 2.0), (0, 0.1), (50, 200)])

base_per_day, mileage_rate, receipt_factor, efficiency_bonus, efficiency_threshold = result.x

print(f'Best efficiency-based formula:')
print(f'  Base: ${base_per_day:.2f} per day')
print(f'  Mileage: ${mileage_rate:.3f} per mile')
print(f'  Receipts: {receipt_factor:.3f} * receipt amount')
print(f'  Efficiency bonus: {efficiency_bonus:.4f} per mile/day above {efficiency_threshold:.1f}')

# Calculate predictions
predicted = np.zeros(len(reimbursements))
for i in range(len(reimbursements)):
    mpd = miles_per_day[i]
    base = base_per_day * trip_days[i]
    mileage = mileage_rate * miles[i]
    receipt_comp = receipt_factor * receipts[i]
    
    if mpd > efficiency_threshold:
        efficiency_mult = 1 + efficiency_bonus * (mpd - efficiency_threshold) / 100
    else:
        efficiency_mult = 1
        
    predicted[i] = (base + mileage + receipt_comp) * efficiency_mult

rmse = np.sqrt(np.mean((predicted - reimbursements) ** 2))
r_squared = 1 - np.sum((predicted - reimbursements)**2) / np.sum((reimbursements - np.mean(reimbursements))**2)

print(f'\nModel performance:')
print(f'  RMSE: {rmse:.3f}')
print(f'  R²: {r_squared:.4f}')

# Check for exact matches
exact_matches = np.sum(np.abs(predicted - reimbursements) < 0.01)
print(f'  Exact matches (within $0.01): {exact_matches}')

print('\n=== TESTING SIMPLER MODELS ===')

# Test a model with caps
def capped_formula(params):
    base_per_day, mileage_rate, receipt_factor, daily_cap = params
    
    predicted = base_per_day * trip_days + mileage_rate * miles + receipt_factor * receipts
    # Apply daily cap
    daily_amount = predicted / trip_days
    capped_daily = np.minimum(daily_amount, daily_cap)
    predicted_capped = capped_daily * trip_days
    
    return np.sum((predicted_capped - reimbursements) ** 2)

result2 = minimize(capped_formula, [100, 0.56, 0.5, 300],
                  bounds=[(10, 200), (0.1, 1.0), (0.1, 2.0), (100, 1000)])

base2, mileage2, receipt2, cap2 = result2.x
predicted2 = base2 * trip_days + mileage2 * miles + receipt2 * receipts
daily_amount2 = predicted2 / trip_days
capped_daily2 = np.minimum(daily_amount2, cap2)
predicted2_capped = capped_daily2 * trip_days

rmse2 = np.sqrt(np.mean((predicted2_capped - reimbursements) ** 2))
r_squared2 = 1 - np.sum((predicted2_capped - reimbursements)**2) / np.sum((reimbursements - np.mean(reimbursements))**2)

print(f'\nCapped formula:')
print(f'  ${base2:.2f}/day + ${mileage2:.3f}/mile + {receipt2:.3f}*receipts, capped at ${cap2:.0f}/day')
print(f'  RMSE: {rmse2:.3f}')
print(f'  R²: {r_squared2:.4f}')

exact_matches2 = np.sum(np.abs(predicted2_capped - reimbursements) < 0.01)
print(f'  Exact matches: {exact_matches2}')