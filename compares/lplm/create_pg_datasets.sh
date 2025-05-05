#!/bin/bash

TARGET_DIR="../../pg_exp/columns" 

for dir in "$TARGET_DIR"/*; do
  # filename=$(basename "$file")
  # fname="${filename%.*}"
  if [ -d "$dir" ]; then
    foldername=$(basename "$dir")
  python compute_ground_truth_end_to_end.py "$foldername"
  fi
done