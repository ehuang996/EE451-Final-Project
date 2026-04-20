#!/bin/bash

# Input file
INPUT="cs_go_winner_data.csv"
TRAIN_OUT="cs_go_winner_data_train.csv"
TEST_OUT="cs_go_winner_data_test.csv"

# 1. Extract the header to a temporary file
head -n 1 "$INPUT" > header.txt

# 2. Get all data except the header, shuffle it, and store in a temp file
tail -n +2 "$INPUT" | shuf > shuffled_data.tmp

# 3. Calculate the split point (80%)
TOTAL_ROWS=$(wc -l < shuffled_data.tmp)
TRAIN_COUNT=$(( TOTAL_ROWS * 80 / 100 ))

# 4. Create the Train set (Header + first 80%)
cat header.txt > "$TRAIN_OUT"
head -n "$TRAIN_COUNT" shuffled_data.tmp >> "$TRAIN_OUT"

# 5. Create the Test set (Header + remaining 20%)
cat header.txt > "$TEST_OUT"
tail -n +$(( TRAIN_COUNT + 1 )) shuffled_data.tmp >> "$TEST_OUT"

# 6. Clean up
rm header.txt shuffled_data.tmp

echo "Done! Created $TRAIN_OUT and $TEST_OUT"
echo "Total data rows: $TOTAL_ROWS (Train: $TRAIN_COUNT, Test: $(( TOTAL_ROWS - TRAIN_COUNT )))"