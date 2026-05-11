#!/bin/bash
# run_unsupervised_methods.sh
# Execute both unsupervised/semi-supervised methods in sequence

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Running Unsupervised & Semi-Supervised Methods"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check that data exists
if [ ! -d "data/processed" ]; then
    echo "❌ ERROR: data/processed directory not found!"
    echo "Please run: python3 src/preprocessing.py"
    exit 1
fi

echo "✓ Data directory found"
echo ""

# Step 1: K-Means
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Running K-Means Clustering"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "src/unsupervised_kmeans.py" ]; then
    echo "❌ ERROR: src/unsupervised_kmeans.py not found!"
    exit 1
fi

python3 src/unsupervised_kmeans.py

if [ $? -ne 0 ]; then
    echo "❌ K-Means failed!"
    exit 1
fi

echo ""
echo "✓ K-Means completed successfully"
echo ""

# Step 2: Label Propagation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Running Label Propagation & Self-Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "src/semi_supervised_label_prop.py" ]; then
    echo "❌ ERROR: src/semi_supervised_label_prop.py not found!"
    exit 1
fi

python3 src/semi_supervised_label_prop.py

if [ $? -ne 0 ]; then
    echo "❌ Label Propagation failed!"
    exit 1
fi

echo ""
echo "✓ Label Propagation completed successfully"
echo ""

# Verify outputs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VERIFYING OUTPUT FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_DIR="models/model_a/unsupervised"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: $OUTPUT_DIR not created!"
    exit 1
fi

# Check K-Means outputs
echo "K-Means outputs:"
for file in kmeans_results.json kmeans_labels_test.npy kmeans_pca_visualization.png; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        size=$(du -h "$OUTPUT_DIR/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ❌ $file (missing)"
    fi
done

echo ""
echo "Label Propagation outputs:"
for file in label_propagation_results.json label_propagation_comparison.png; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        size=$(du -h "$OUTPUT_DIR/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ❌ $file (missing)"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ ALL METHODS COMPLETED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
echo "📝 Next Step: Fill FINAL_REPORT.md section 3.2 with these results:"
echo ""
echo "   1. Open FINAL_REPORT.md"
echo "   2. Find section '3.2 Unsupervised/Semi-Supervised Results'"
echo "   3. Fill in metrics from these JSON files:"
echo "      - $OUTPUT_DIR/kmeans_results.json"
echo "      - $OUTPUT_DIR/label_propagation_results.json"
echo ""
echo "⏱️  Estimated time to fill report: 30-60 minutes"
echo ""
echo "🎯 Then write section 6 (Conclusion) explaining your findings."
echo ""
