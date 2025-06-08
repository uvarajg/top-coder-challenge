#!/usr/bin/env python3

import sys
import json
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Pre-calculated model weights and parameters for instant prediction
BUSINESS_RULES = {
    'base_rate': 85.0,
    'per_day': 45.0,
    'per_mile': 0.42,
    'per_receipt_dollar': 0.28,
    'efficiency_bonus': {180: 35.0, 200: 20.0},
    'five_day_bonus': 25.0,
    'vacation_penalty': 0.88,
    'rounding_bonus': 15.0,
    'small_receipt_penalty': -25.0
}

def ultra_fast_predict(days, miles, receipts):
    """Ultra-fast prediction using business rules and ML hybrid"""
    
    # Convert inputs to float
    days = float(days)
    miles = float(miles)
    receipts = float(receipts)
    
    # Calculate base efficiency and spending
    efficiency = miles / max(days, 1)
    daily_spending = receipts / max(days, 1)
    
    # Start with base calculation
    prediction = BUSINESS_RULES['base_rate']
    prediction += days * BUSINESS_RULES['per_day']
    prediction += miles * BUSINESS_RULES['per_mile']
    prediction += receipts * BUSINESS_RULES['per_receipt_dollar']
    
    # Apply efficiency bonuses
    if 180 <= efficiency <= 220:
        prediction += BUSINESS_RULES['efficiency_bonus'][180]
    elif efficiency >= 200:
        prediction += BUSINESS_RULES['efficiency_bonus'][200]
    elif efficiency < 20 and days > 1:
        prediction -= 20.0
    
    # Kevin's sweet spot bonus
    if days == 5 and efficiency >= 180 and daily_spending < 100:
        prediction *= 1.12
    
    # Five day bonus
    if days == 5:
        prediction += BUSINESS_RULES['five_day_bonus']
    
    # Vacation penalty
    if days >= 8 and daily_spending > 150:
        prediction *= BUSINESS_RULES['vacation_penalty']
    
    # Lisa's rounding quirk
    if receipts > 0:
        cents = int((receipts * 100) % 100)
        if cents == 49 or cents == 99:
            prediction += BUSINESS_RULES['rounding_bonus']
    
    # Small receipt penalty
    if 0 < receipts < 30 and days > 1:
        prediction += BUSINESS_RULES['small_receipt_penalty']
    
    # Trip length adjustments
    if days <= 2:
        prediction += 35.0  # Short trip bonus
    elif days >= 12:
        penalty = (days - 11) * 15
        prediction -= penalty
    
    # Apply reasonable bounds
    min_reasonable = days * 40.0
    max_reasonable = days * 450 + miles * 1.5 + receipts * 1.0
    prediction = max(min_reasonable, min(prediction, max_reasonable))
    
    return round(max(0, prediction), 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ultra_fast_system.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = ultra_fast_predict(trip_duration, miles, receipts)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)