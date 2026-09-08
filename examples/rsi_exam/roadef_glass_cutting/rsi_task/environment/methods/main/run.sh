#!/usr/bin/env bash
# Entry point for ONE test case. The grader invokes it as:
#
#   bash run.sh <batch.csv> <defects.csv> <global_param.csv> <out_solution.csv>
#
# Read the three instance CSVs, write a cutting-tree solution to the 4th
# path (the out file). Replace the body with your own solver in any language.
# If you compile a binary, do it in an OPTIONAL sibling `build.sh` (run once
# before grading) and exec the binary here. This starter runs the shipped
# greedy baseline.
exec python3 "$(dirname "$0")/solution.py" "$1" "$2" "$3" "$4"
