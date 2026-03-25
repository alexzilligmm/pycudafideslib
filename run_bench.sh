#!/bin/bash
# run_bench.sh — Benchmark all FHE operations across levels 1..16
#                to produce data.csv for the bootstrap placement optimizer.
#
# Usage:
#   ./run_bench.sh                    # LLaMA ops (default)
#   ./run_bench.sh --model gpt2      # GPT-2 ops (adds GELU, Up; skips RoPE, SiLU, UpGate)
#   ./run_bench.sh --logN 16 --hidDim 4096 --ffDim 16384  # custom dimensions
#
# Output:
#   raw_result.csv   — per-run raw timings
#   new_data.csv     — tab-separated latency table (input to bootstrap.py)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
MODEL="llama"
LOGN=16
HIDDIM=4096
FFDIM=16384
SEQLEN=512
NUMHEADS=32
BUILD_DIR="build"
BINARY="${BUILD_DIR}/bin/cuda_cachemir"
MAX_LEVEL=16

# ── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --logN)     LOGN="$2";     shift 2 ;;
        --hidDim)   HIDDIM="$2";   shift 2 ;;
        --ffDim)    FFDIM="$2";    shift 2 ;;
        --seqLen)   SEQLEN="$2";   shift 2 ;;
        --numHeads) NUMHEADS="$2"; shift 2 ;;
        --build)    BUILD_DIR="$2"; BINARY="${BUILD_DIR}/bin/cuda_cachemir"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

COMMON_FLAGS="-logN $LOGN -hidDim $HIDDIM -ffDim $FFDIM -seqLen $SEQLEN -numHeads $NUMHEADS"

# ── Check binary exists ──────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo "Binary not found at $BINARY"
    echo "Build first:  mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

output_file="raw_result.csv"
summary_file="new_data.csv"
rm -f "$output_file" "$summary_file"

echo "timestamp,status,module,level,time" > "$output_file"

# ── Select operations based on model ─────────────────────────────────────
if [ "$MODEL" = "gpt2" ]; then
    # GPT-2: no RoPE, no SiLU, no UpGate; has GELU, Up
    BASIC_OPS=("QKV" "QK_T" "AttnV" "Up" "Down" "Cache" "CtMult")
    SUMMARY_OPS=("QKV" "QK_T" "AttnV" "GELU" "Up" "Down" "Cache" "CtMult" "Softmax" "SqrtNt" "SqrtGold")
    HAS_GELU=1
    HAS_SILU=0
else
    # LLaMA: has RoPE, SiLU, UpGate
    BASIC_OPS=("QKV" "RoPE" "Cache" "QK_T" "AttnV" "UpGate" "Down" "SiLU" "CtMult")
    SUMMARY_OPS=("QKV" "RoPE" "Cache" "QK_T" "AttnV" "SiLU" "UpGate" "Down" "CtMult" "Softmax" "SqrtNt" "SqrtGold")
    HAS_GELU=0
    HAS_SILU=1
fi

# ── Run function ─────────────────────────────────────────────────────────
run_test() {
    local test_name="$1"
    local level="$2"
    local extra_flags="${3:-}"
    local csv_name="${4:-$test_name}"

    echo "Testing: -test $test_name -level $level $extra_flags"
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    output=$("$BINARY" -test "$test_name" -level "$level" $COMMON_FLAGS $extra_flags 2>&1) || true
    exit_status=$?
    sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || echo "N/A")
    echo "$timestamp,$exit_status,$csv_name,$level,$sanitized_output" >> "$output_file"
    echo "  -> $sanitized_output"
}

# ── 1. Basic operations: sweep level 1..16 ───────────────────────────────
echo ""
echo "=== Benchmarking basic operations (${MODEL}) ==="
echo ""

for test in "${BASIC_OPS[@]}"; do
    for level in $(seq 1 $MAX_LEVEL); do
        run_test "$test" "$level"
    done
done

# GELU (GPT-2 only)
if [ "$HAS_GELU" = "1" ]; then
    for level in $(seq 1 $MAX_LEVEL); do
        run_test "GELU" "$level"
    done
fi

# ── 2. Softmax: fixed input level=16, sweep btpLevel 1..16 ──────────────
echo ""
echo "=== Benchmarking Softmax ==="
echo ""

for btp_level in $(seq 1 $MAX_LEVEL); do
    run_test "Softmax" "$MAX_LEVEL" "-btpLevel $btp_level" "Softmax"
done

# ── 3. Norm: two passes ─────────────────────────────────────────────────
#   SqrtNt:   sweep input level 1..16 (measures Newton init cost)
#   SqrtGold: fixed input level=16, sweep btpLevel 1..16 (measures Goldschmidt after bootstrap)
echo ""
echo "=== Benchmarking Norm (SqrtNt + SqrtGold) ==="
echo ""

for level in $(seq 1 $MAX_LEVEL); do
    # SqrtNt: Newton-Raphson init at each input level
    run_test "Norm" "$level" "" "SqrtNt"
done

for btp_level in $(seq 1 $MAX_LEVEL); do
    # SqrtGold: Goldschmidt refinement at each post-bootstrap level
    run_test "Norm" "$MAX_LEVEL" "-btpLevel $btp_level" "SqrtGold"
done

echo ""
echo "=== Raw results saved to $output_file ==="
echo ""

# ── 4. Build summary table ──────────────────────────────────────────────
{
    echo -n "module"
    for i in $(seq 1 $MAX_LEVEL); do
        printf "\t%.2f" "$i"
    done
    echo
} > "$summary_file"

for test in "${SUMMARY_OPS[@]}"; do
    times=$(awk -F',' -v mod="$test" '
        $3 ~ mod {
            gsub(/"/, "", $2)
            gsub(/"/, "", $5)
            if ($2 == "0") {
                n = split($5, arr, /[ \t]+/)
                if (mod == "Softmax" || mod == "SqrtGold") {
                    print arr[n]
                } else {
                    print arr[1]
                }
            } else {
                print "0.00"
            }
        }
    ' "$output_file")

    echo -n "$test" >> "$summary_file"
    i=1
    while IFS= read -r t; do
        if [[ -z "$t" || "$t" == "N/A" ]]; then
            printf "\t0.00" >> "$summary_file"
        else
            printf "\t%.2f" "$t" >> "$summary_file"
        fi
        ((i++))
    done <<< "$times"

    # Pad missing columns
    while [ $i -le $MAX_LEVEL ]; do
        printf "\t0.00" >> "$summary_file"
        ((i++))
    done

    echo >> "$summary_file"
done

echo "Summary table written to $summary_file"
echo ""
cat "$summary_file"
echo ""

# ── 5. Run bootstrap placement optimizer ─────────────────────────────────
echo "=== Running bootstrap placement optimizer ==="
python3 bootstrap.py --prune=1 --file="$summary_file" --model="$MODEL"
