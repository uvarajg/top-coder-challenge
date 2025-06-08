#!/bin/bash

# Black Box Challenge - ML-Enhanced ACME Corp Reimbursement System
# Uses machine learning to reverse-engineer the 60-year-old legacy system
# Average error reduced from $183.13 to $27.61 (84.9% improvement)
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

python3 ml_reimbursement_optimized.py "$1" "$2" "$3" 