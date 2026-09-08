#!/usr/bin/env bash
# Entry point for one test case: read an instance on stdin, write the
# F/R/L/S/M/P button sequence on stdout. The grader invokes this once per
# instance as:  bash run.sh < instance.txt > out.txt
#
# Replace the body with your own solver in any language. If you compile a
# binary, do it in an OPTIONAL sibling `build.sh` (run once before grading)
# and exec the binary here. This starter runs the shipped greedy baseline.
exec python3 "$(dirname "$0")/solution.py"
