#!/bin/bash

# Black Box Challenge - ML-Enhanced ACME Corp Reimbursement System
# Uses machine learning to reverse-engineer the 60-year-old legacy system
# Usage: ./run_ml.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

python3 ml_reimbursement.py "$1" "$2" "$3"