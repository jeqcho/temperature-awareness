#!/bin/bash
echo "=== Experiment Progress Monitor ==="
echo ""

# Check if prefixed experiment is running
if pgrep -f "run_experiment.py" > /dev/null; then
    echo "Prefixed Experiment: RUNNING"
    tail -1 /home/ubuntu/temperature-awareness/results_prefixed/experiment.log 2>/dev/null | grep -o 'Running experiments:.*%' | head -1 || echo "  (waiting for progress...)"
else
    echo "Prefixed Experiment: COMPLETED"
    if [ -f "/home/ubuntu/temperature-awareness/results_prefixed/experiment_results.json" ]; then
        echo "  Results file exists"
    fi
fi

echo ""

# Check if classification experiment is running
if pgrep -f "run_transcript_classification.py" > /dev/null; then
    echo "Classification Experiment: RUNNING"
    tail -1 /home/ubuntu/temperature-awareness/results_classification/classification.log 2>/dev/null | grep -o 'Classifying transcripts:.*%' | head -1 || echo "  (waiting for progress...)"
else
    echo "Classification Experiment: COMPLETED"
    if [ -f "/home/ubuntu/temperature-awareness/results_classification/classification_results.json" ]; then
        echo "  Results file exists"
    fi
fi
