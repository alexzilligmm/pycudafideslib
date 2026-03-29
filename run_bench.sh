#!/bin/bash
# run_bench.sh — Benchmark all FHE operations across levels 1..16
#                to produce data.csv for the bootstrap placement optimizer.
#
# Usage:
#   ./run_bench.sh                    # LLaMA ops (default)
#   ./run_bench.sh --model gpt2      # GPT-2 ops (adds GELU, Up; skips RoPE, SiLU, UpGate)
#   ./run_bench.sh --logN 16 --hidDim 4096 --ffDim 16384  # custom dimensions
#   ./run_bench.sh --test-timeout 120 # kill each benchmark test after 120s (0 = disabled)
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
PARALLEL="true"
MAX_LEVEL=16
OUTPUT_PREFIX=""
TEST_TIMEOUT_SECS="${TEST_TIMEOUT_SECS:-0}"

# ── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --logN)     LOGN="$2";     shift 2 ;;
        --hidDim)   HIDDIM="$2";   shift 2 ;;
        --ffDim)    FFDIM="$2";    shift 2 ;;
        --seqLen)   SEQLEN="$2";   shift 2 ;;
        --numHeads) NUMHEADS="$2"; shift 2 ;;
        --maxLevel) MAX_LEVEL="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --test-timeout) TEST_TIMEOUT_SECS="$2"; shift 2 ;;
        --build)    BUILD_DIR="$2"; BINARY="${BUILD_DIR}/bin/cuda_cachemir"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

for kv in \
    "logN:$LOGN" \
    "hidDim:$HIDDIM" \
    "ffDim:$FFDIM" \
    "seqLen:$SEQLEN" \
    "numHeads:$NUMHEADS" \
    "maxLevel:$MAX_LEVEL" \
    "testTimeoutSecs:$TEST_TIMEOUT_SECS"; do
    key="${kv%%:*}"
    val="${kv#*:}"
    if [[ ! "$val" =~ ^[0-9]+$ ]]; then
        echo "Invalid value for --$key: '$val' (expected integer)"
        exit 1
    fi
done

COMMON_FLAGS="-model $MODEL -logN $LOGN -hidDim $HIDDIM -ffDim $FFDIM -seqLen $SEQLEN -numHeads $NUMHEADS -parallel $PARALLEL"

if [ ! -x "$BINARY" ]; then
    echo "Binary not found at $BINARY"
    echo "Build first:  mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

if [[ -n "$OUTPUT_PREFIX" ]]; then
    output_file="${OUTPUT_PREFIX}_raw_result.csv"
    summary_file="${OUTPUT_PREFIX}_new_data.csv"
else
    output_file="raw_result.csv"
    summary_file="new_data.csv"
fi
rm -f "$output_file" "$summary_file"

echo "timestamp,status,module,level,time" > "$output_file"

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
    set +e
    if command -v timeout >/dev/null 2>&1 && [[ "$TEST_TIMEOUT_SECS" -gt 0 ]]; then
        output=$(timeout "${TEST_TIMEOUT_SECS}s" "$BINARY" -test "$test_name" -level "$level" $COMMON_FLAGS $extra_flags 2>&1)
    else
        output=$("$BINARY" -test "$test_name" -level "$level" $COMMON_FLAGS $extra_flags 2>&1)
    fi
    exit_status=$?
    set -e
    sanitized_output=$(echo "$output" | grep -oP 'Consumed\s+\K[0-9.]+(?=\s+seconds)' | tr '\n' ' ' | sed 's/[[:space:]]*$//' || true)
    if [[ -z "$sanitized_output" ]]; then
        sanitized_output="N/A"
    fi

    error_hint=""
    if [[ "$exit_status" -ne 0 || "$sanitized_output" == "N/A" ]]; then
        error_hint=$(echo "$output" | awk '
            /what\(\):/ { print; exit }
            /terminate called/ { print; exit }
            /Invalid|Unknown|Exception|error|Error/ { print; exit }
        ')
        if [[ -z "$error_hint" ]]; then
            error_hint=$(echo "$output" | head -n 1)
        fi
    fi

    echo "$timestamp,$exit_status,$csv_name,$level,$sanitized_output" >> "$output_file"
    if [[ -n "$error_hint" ]]; then
        echo "  -> $sanitized_output (exit=$exit_status; ${error_hint})"
    else
        echo "  -> $sanitized_output"
    fi
}

echo ""
echo "=== Benchmarking basic operations (${MODEL}) ==="
echo ""

for test in "${BASIC_OPS[@]}"; do
    for level in $(seq 1 $MAX_LEVEL); do
        run_test "$test" "$level"
    done
done

if [ "$HAS_GELU" = "1" ]; then
    for level in $(seq 1 $MAX_LEVEL); do
        run_test "GELU" "$level"
    done
fi

echo ""
echo "=== Benchmarking Bootstrap ==="
echo ""

# Run bootstrap at a mid-range level to get representative latency
BOOT_LAT=""
for level in $(seq 1 $MAX_LEVEL); do
    run_test "Bootstrap" "$level"
done
# Extract the median bootstrap latency from raw results
BOOT_LAT=$(python3 -c "
import csv, statistics
vals = []
with open('$output_file') as f:
    for r in csv.reader(f):
        if len(r)>=5 and r[2]=='Bootstrap' and r[1]=='0':
            try: vals.append(float(r[4]))
            except: pass
if vals: print(f'{statistics.median(vals):.6f}')
")
echo "Bootstrap latency (median): ${BOOT_LAT:-N/A} seconds"

echo ""
echo "=== Benchmarking Softmax ==="
echo ""

for btp_level in $(seq 1 $MAX_LEVEL); do
    run_test "Softmax" "$MAX_LEVEL" "-btpLevel $btp_level" "Softmax"
done

echo ""
echo "=== Benchmarking Norm (SqrtNt + SqrtGold) ==="
echo ""

for level in $(seq 1 $MAX_LEVEL); do
    run_test "Norm" "$level" "" "SqrtNt"
done

for btp_level in $(seq 1 $MAX_LEVEL); do
    run_test "Norm" "$MAX_LEVEL" "-btpLevel $btp_level" "SqrtGold"
done

echo ""
echo "=== Raw results saved to $output_file ==="
echo ""

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
                print "0"
            }
        }
    ' "$output_file")

    echo -n "$test" >> "$summary_file"
    i=1
    while IFS= read -r t; do
        if [[ -z "$t" || "$t" == "N/A" ]]; then
            printf "\t0" >> "$summary_file"
        else
            printf "\t%.6f" "$t" >> "$summary_file"
        fi
        ((i++))
    done <<< "$times"

    while [ $i -le $MAX_LEVEL ]; do
        printf "\t0" >> "$summary_file"
        ((i++))
    done

    echo >> "$summary_file"
done

echo "Summary table written to $summary_file"
echo ""
cat "$summary_file"
echo ""

echo "=== Running bootstrap placement optimizer ==="
BOOT_OPT=""
if [[ -n "$BOOT_LAT" && "$BOOT_LAT" != "N/A" ]]; then
    BOOT_OPT="--boot-lat $BOOT_LAT"
    echo "Using measured bootstrap latency: $BOOT_LAT s"
fi
# Pass --uniform if set in environment to force same BTP config for all blocks
UNIFORM_OPT=""
if [[ "${UNIFORM_PLACEMENT:-0}" == "1" ]]; then
    UNIFORM_OPT="--uniform"
    echo "Uniform mode: BTP forced at start of every decoder block"
fi
uv run python bootstrap.py --prune=1 --file="$summary_file" --model="$MODEL" --logN="$LOGN" --max-level="$MAX_LEVEL" $BOOT_OPT $UNIFORM_OPT || echo "bootstrap.py failed (non-fatal)"
