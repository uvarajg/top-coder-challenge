#!/usr/bin/env python3

import json
import numpy as np
from ml_reimbursement_silent import SilentReimbursementMLModel
import sys

def generate_private_results():
    """Generate results for all private cases efficiently"""
    
    print("🧾 Black Box Challenge - Generating ML Private Results")
    print("====================================================")
    print()
    print("📊 Loading and training ML model...")
    
    # Train model once
    model = SilentReimbursementMLModel()
    print("✅ Model trained successfully!")
    
    print("📊 Processing private test cases...")
    
    # Load private cases
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    total_cases = len(private_cases)
    print(f"📈 Processing {total_cases} test cases...")
    
    results = []
    successful = 0
    errors = 0
    
    for i, case in enumerate(private_cases):
        if (i + 1) % 500 == 0:
            print(f"Progress: {i + 1}/{total_cases} cases processed...")
        
        try:
            days = case['trip_duration_days']
            miles = case['miles_traveled'] 
            receipts = case['total_receipts_amount']
            
            # Make prediction
            prediction = model.predict(days, miles, receipts)
            results.append(str(prediction))
            successful += 1
            
        except Exception as e:
            results.append("ERROR")
            errors += 1
            if errors <= 5:  # Only print first few errors
                print(f"Error in case {i+1}: {e}")
    
    # Write results to file
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"✅ Results generated successfully!")
    print(f"📄 Output saved to private_results.txt")
    print(f"📊 Successful predictions: {successful}/{total_cases}")
    if errors > 0:
        print(f"⚠️  Errors encountered: {errors}")
    
    return successful, errors

if __name__ == "__main__":
    generate_private_results()