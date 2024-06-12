#!/bin/bash

# Check if eval_address is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <eval_address>"
  exit 1
fi

EVAL_ADDRESS=$1
DECISION_TYPES=("majority" "judge")

for decision in "${DECISION_TYPES[@]}"
do
  python evaluate.py \
         --eval_address "$EVAL_ADDRESS" \
         --decision "$decision"
done