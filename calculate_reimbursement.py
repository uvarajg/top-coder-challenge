#!/usr/bin/env python3
import sys
import math

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Reverse-engineered ACME Corp travel reimbursement calculation
    Simple linear formula with adjustments based on data analysis
    """
    
    # Convert inputs to appropriate types
    days = int(float(trip_duration_days))
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    # Best-fit linear formula from data analysis: $63.19/day + $0.578/mile + 0.434 × receipts
    # But tuned based on efficiency patterns
    
    efficiency = miles / days if days > 0 else 0
    
    # Use different coefficients based on efficiency (from data analysis)
    if efficiency >= 50:  # High efficiency trips (707 cases)
        day_rate = 57.69
        mile_rate = 0.645
        receipt_rate = 0.421
    else:  # Low efficiency trips (293 cases)
        day_rate = 69.89
        mile_rate = 0.270
        receipt_rate = 0.455
    
    # Base calculation
    reimbursement = day_rate * days + mile_rate * miles + receipt_rate * receipts
    
    # === SPECIFIC ADJUSTMENTS FROM INTERVIEWS ===
    
    # 5-day bonus (Lisa and Kevin confirmed)
    if days == 5:
        reimbursement += 15
    
    # Small receipt penalty (multiple people mentioned)
    if 0 < receipts < 30 and days > 1:
        reimbursement -= 10
    
    # Trip length effects
    if days <= 2:
        reimbursement += 20  # Short trip bonus
    elif days >= 10:
        # Long trip penalty gets stronger
        penalty = (days - 9) * 12
        reimbursement -= penalty
    
    # Kevin's efficiency sweet spot (180-220 miles/day)
    if 180 <= efficiency <= 220:
        reimbursement += 25
    elif efficiency >= 300:  # Very high efficiency bonus
        reimbursement += 15
    elif efficiency < 20 and days > 1:  # Low efficiency penalty
        reimbursement -= 15
    
    # Lisa's rounding quirk for receipts ending in .49 or .99
    if receipts > 0:
        cents = int((receipts * 100) % 100)
        if cents == 49 or cents == 99:
            reimbursement += 6
    
    # Some controlled variation for unexplained variance
    variation_seed = hash((days, int(miles), int(receipts * 100))) % 100
    if variation_seed < 10:
        reimbursement += 10
    elif variation_seed > 90:
        reimbursement -= 10
    
    # Round to 2 decimal places as required
    return round(max(0, reimbursement), 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration = sys.argv[1]
        miles = sys.argv[2] 
        receipts = sys.argv[3]
        
        result = calculate_reimbursement(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)