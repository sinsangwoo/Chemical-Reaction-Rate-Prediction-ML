#!/bin/bash

# Complete experimental pipeline for Phase 6.1
# Runs benchmark, statistical analysis, and ablation studies

set -e  # Exit on error

echo "========================================"
echo "  Phase 6.1: Benchmark Experiments"
echo "========================================"
echo ""

# Create results directory
mkdir -p experiments/results

# Step 1: Run benchmark
echo "[1/3] Running benchmark experiments..."
echo "--------------------------------------"
python experiments/benchmark.py

if [ $? -eq 0 ]; then
    echo "✓ Benchmark complete!"
else
    echo "✗ Benchmark failed!"
    exit 1
fi

echo ""

# Step 2: Statistical analysis
echo "[2/3] Running statistical analysis..."
echo "--------------------------------------"

# Find latest results file
RESULTS_FILE=$(ls -t experiments/results/benchmark_results_*.csv 2>/dev/null | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo "✗ No results file found!"
    exit 1
fi

echo "Using results: $RESULTS_FILE"
python experiments/statistical_analysis.py "$RESULTS_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Statistical analysis complete!"
else
    echo "✗ Statistical analysis failed!"
    exit 1
fi

echo ""

# Step 3: Ablation study
echo "[3/3] Running ablation study..."
echo "--------------------------------------"
python experiments/ablation_study.py

if [ $? -eq 0 ]; then
    echo "✓ Ablation study complete!"
else
    echo "✗ Ablation study failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "  All experiments completed!"
echo "========================================"
echo ""
echo "Results saved in: experiments/results/"
echo ""
echo "Generated files:"
ls -lh experiments/results/ | tail -n +2

echo ""
echo "Next steps:"
echo "  1. Review summary: cat experiments/results/summary_report_*.txt"
echo "  2. View figures: open experiments/results/*.png"
echo "  3. Check statistics: cat experiments/results/statistical_tests.csv"
echo "  4. Use LaTeX tables in paper: experiments/results/table_*.tex"
echo ""
