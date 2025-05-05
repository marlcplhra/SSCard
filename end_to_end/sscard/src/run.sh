#!/bin/bash

TARGET_DIR="../../columns" 

for file in "$TARGET_DIR"/*; do
  filename=$(basename "$file")
  fname="${filename%.*}"
  python fm_build.py --dname="$fname"
done
