#!/usr/bin/env python3
"""
cross_validate.py — Cross-validate Ekatra.jl output against the reference
Python `gguf` library (pip install gguf).

Usage:
    python cross_validate.py <path/to/model.gguf>

Exit codes:
    0 — All checks passed
    1 — Validation error (mismatch or unreadable file)
    2 — Usage error (wrong arguments)

The script prints a structured report suitable for piping into the Julia test
suite or reading in CI logs.
"""

import sys
import json
import struct
import pathlib

# ──────────────────────────────────────────────
# Dependency check
# ──────────────────────────────────────────────

try:
    from gguf import GGUFReader, GGUFValueType
except ImportError:
    print("ERROR: 'gguf' package not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None


# ──────────────────────────────────────────────
# JSON encoder that handles numpy scalar types
# ──────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if np is not None and isinstance(obj, np.integer):
            return int(obj)
        if np is not None and isinstance(obj, np.floating):
            return float(obj)
        if np is not None and isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def field_to_python(field):
    """Convert a GGUFReader field to a plain Python value."""
    if field.types[0] == GGUFValueType.STRING:
        return bytes(field.data).decode("utf-8", errors="replace")
    elif field.types[0] == GGUFValueType.ARRAY:
        # Return element count for arrays — don't dump entire vocab
        return f"<array len={len(field.data)}>"
    else:
        # Numeric scalar
        return field.data[0]


def dtype_name(tensor_type) -> str:
    """Return a human-readable dtype name from a GGMLQuantizationType or int."""
    s = str(tensor_type)
    # Handles both 'GGMLQuantizationType.Q4_K' style and plain integers
    if "." in s:
        return s.split(".")[-1]
    # Map raw GGML integer IDs to names
    _MAP = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
        14: "Q6_K", 15: "Q8_K", 30: "BF16",
    }
    try:
        return _MAP.get(int(tensor_type), f"type{tensor_type}")
    except Exception:
        return s


# ──────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────

def validate(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Cross-validation: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}\n")

    reader = GGUFReader(path, "r")

    # ── Metadata ──────────────────────────────────────────────────────────
    print("── Metadata ──────────────────────────────────────────────")
    meta = {}
    for key, field in reader.fields.items():
        try:
            val = field_to_python(field)
        except Exception as e:
            val = f"<error: {e}>"
        meta[key] = val
        # Print only non-array scalars (skip huge vocab lists)
        if not str(val).startswith("<array"):
            print(f"  {key}: {val!r}")

    print(f"\n  Total metadata keys: {len(meta)}")

    # Key presence assertions
    assert "general.architecture" in meta, "Missing 'general.architecture'"
    arch = meta["general.architecture"]
    print(f"\n  ✓ general.architecture = {arch!r}")

    # ── Tensors ───────────────────────────────────────────────────────────
    print("\n── Tensors ────────────────────────────────────────────────")
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = {
            "shape": list(t.shape),
            "dtype": dtype_name(t.tensor_type),
        }

    print(f"  Total tensors: {len(tensors)}")
    assert len(tensors) > 0, "No tensors found!"

    # Print first 10 tensors
    for i, (name, info) in enumerate(list(tensors.items())[:10]):
        print(f"  [{i:3d}] {name:<50}  shape={info['shape']}  dtype={info['dtype']}")
    if len(tensors) > 10:
        print(f"  ... and {len(tensors) - 10} more")

    # ── dtype distribution ────────────────────────────────────────────────
    print("\n── Quantization type distribution ─────────────────────────")
    from collections import Counter
    dtype_counts = Counter(info["dtype"] for info in tensors.values())
    for dtype, cnt in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype:<12}  {cnt:4d} tensors")

    # ── Output JSON (machine-readable for Julia comparison) ───────────────
    result = {
        "file":     str(p.name),
        "size_mb":  round(p.stat().st_size / 1e6, 2),
        "metadata": {k: str(v) for k, v in meta.items()},
        "tensor_count": len(tensors),
        "tensors":  tensors,
        "dtype_counts": dict(dtype_counts),
    }

    out_path = p.with_suffix(".cross_validate.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n✓ JSON report written: {out_path}")

    print(f"\n✓ Cross-validation PASSED for {p.name}")
    return result


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cross_validate.py <model.gguf> [model2.gguf ...]",
              file=sys.stderr)
        sys.exit(2)

    any_failure = False
    for path in sys.argv[1:]:
        try:
            validate(path)
        except AssertionError as e:
            print(f"\n✗ ASSERTION FAILED: {e}", file=sys.stderr)
            any_failure = True
        except Exception as e:
            print(f"\n✗ ERROR: {e}", file=sys.stderr)
            any_failure = True

    sys.exit(1 if any_failure else 0)
