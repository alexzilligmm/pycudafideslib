"""
Tokenize text and prepare metadata for FHE GPT-2 inference.

Outputs to --out_dir:
  metadata.json  — token IDs, candidates, CLI args for C++ binary

Usage (text generation):
  uv run python scripts/tokenize_input.py \
      --text "The cat sat on the" \
      --out_dir input_data/generate/

Usage (SST-2 sentiment classification):
  uv run python scripts/tokenize_input.py \
      --glue-sst2 "The movie was absolutely wonderful" \
      --out_dir input_data/classify/

  uv run python scripts/tokenize_input.py \
      --glue-sst2 "This film was terrible and boring" \
      --out_dir input_data/classify/

The script prints the -prompt and -candidates flags to pass directly to
the C++ binary:

  ./build/bin/cuda_cachemir -test Generate -model gpt2 \\
      -weights weights-gpt2/ -logN 16 -hidDim 1024 -ffDim 4096 \\
      -realHidDim 768 -realFfDim 3072 -numLayers 1 -numGen 1 \\
      -prompt <ids>

  ./build/bin/cuda_cachemir -test Classify -model gpt2 \\
      -weights weights-gpt2/ -logN 16 -hidDim 1024 -ffDim 4096 \\
      -realHidDim 768 -realFfDim 3072 -numLayers 12 \\
      -prompt <ids> -candidates <cands>
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text",       type=str, help="Input text for generation")
    parser.add_argument("--glue-sst2",  type=str, dest="glue_sst2",
                        help="SST-2 sentence for sentiment classification")
    parser.add_argument("--out_dir",    type=str, default="input_data")
    args = parser.parse_args()

    if not args.text and not args.glue_sst2:
        parser.print_help()
        sys.exit(1)

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: uv add transformers", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    is_classify = False
    candidates = []

    if args.text:
        text = args.text
    else:
        # SST-2: append a short classification template so the model
        # predicts a sentiment word as the next token.
        text = args.glue_sst2.strip() + " It was"
        is_classify = True
        # GPT-2 BPE tokens for " positive" and " negative"
        candidates = [3967, 4633]

    token_ids  = tokenizer.encode(text)
    tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Text   : {text!r}")
    print(f"Tokens : {list(zip(token_ids, tokens_str))}")
    print()

    meta = {
        "text":            text,
        "token_ids":       token_ids,
        "tokens_str":      tokens_str,
        "is_classify":     is_classify,
        "candidates":      candidates,
        "prompt_ids_csv":  ",".join(map(str, token_ids)),
        "candidates_csv":  ",".join(map(str, candidates)),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    prompt_flag  = f"-prompt {meta['prompt_ids_csv']}"
    cand_flag    = f"-candidates {meta['candidates_csv']}" if candidates else ""
    common_flags = (
        "-model gpt2 -weights weights-gpt2/ "
        "-logN 16 -hidDim 1024 -ffDim 4096 "
        "-realHidDim 768 -realFfDim 3072"
    )

    print(f"Saved metadata to {out_dir}/metadata.json")
    print()
    if is_classify:
        cands_decoded = [tokenizer.decode([c]) for c in candidates]
        print(f"Candidate tokens: {list(zip(candidates, cands_decoded))}")
        print()
        print("C++ classification command:")
        print(f"  $BIN -test Classify {common_flags} -numLayers 12 "
              f"{prompt_flag} {cand_flag}")
    else:
        print("C++ generation command:")
        print(f"  $BIN -test Generate {common_flags} -numLayers 1 -numGen 1 "
              f"{prompt_flag}")

    print()
    print(f"Prompt token IDs   : {token_ids}")
    if candidates:
        print(f"Candidate token IDs: {candidates}")


if __name__ == "__main__":
    main()
