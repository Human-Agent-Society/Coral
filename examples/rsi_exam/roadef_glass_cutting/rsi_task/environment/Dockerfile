FROM python:3.11-slim

# Agent workbench: python3 + g++ for writing/compiling solvers, plus the solution
# checker (built from source under checker_src/) used by the local dev bench to
# validate + score solutions exactly as the grader does.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY checker_src/ /app/checker_src/
RUN g++ -O2 -I/app/checker_src/include /app/checker_src/src/*.cpp -o /app/checker

COPY selfcheck.py /app/
COPY tools/ /app/tools/
COPY methods/ /app/methods/
RUN chmod +x /app/methods/main/run.sh /app/checker
