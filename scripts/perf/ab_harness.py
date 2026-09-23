"""Focused confirmation: rotate the arm order so position effects cancel.

ABBA samples only two orderings, so with 7 arms an arm can sit in a
systematically favourable slot. Here each round rotates the start, and TWO
same-config controls bracket the noise: whatever separates control_a from
control_b is the floor, measured the same way as every candidate.
"""
import json
import os
import statistics
import subprocess
import sys

SC = os.path.dirname(os.path.abspath(__file__))
PY_BIN = os.environ.get("AB_PYTHON", sys.executable)
SECONDS = float(sys.argv[1])
REPS = int(sys.argv[2])
ROUNDS = int(sys.argv[3])

# control_a / control_b run the baseline code; the runner ignores unknown arms.
ARMS = ["control_a", "compiled_forward", "flush_1", "baseline",
        "compiled_flush_1", "control_b"]
results = {a: [] for a in ARMS}
checks = {}

for rnd in range(ROUNDS):
    order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
    if rnd % 2:
        order = order[::-1]
    for arm in order:
        p = subprocess.run([PY_BIN, f"{SC}/ab_arm_runner.py", arm, str(SECONDS), str(REPS)],
                           capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None)
        if line is None:
            print(f"  !! {arm}: {p.stderr.strip()[-200:]}", file=sys.stderr)
            continue
        d = json.loads(line)
        results[arm].append(d["median"])
        checks[arm] = d["checksum"]
        print(f"  r{rnd+1} {arm:18s} {d['median']:.4f} (+/-{d['spread_pct']:.2f}%)",
              file=sys.stderr)

ca = statistics.median(results["control_a"])
cb = statistics.median(results["control_b"])
ctrl = statistics.median(results["control_a"] + results["control_b"])
noise = abs(ca - cb) / ctrl * 100
print(f"## Confirmation — htdemucs {SECONDS:.0f} s, {REPS} reps x {ROUNDS} rotated rounds\n")
print(f"**Noise floor {noise:.2f}%** (control_a {ca:.4f} s vs control_b {cb:.4f} s, "
      f"identical config). Reference = both controls pooled: {ctrl:.4f} s.\n")
print("| arm | median | vs control | verdict | output |")
print("|---|---|---|---|---|")
base_ck = checks.get("control_a")
for a in ARMS:
    if not results[a]:
        continue
    m = statistics.median(results[a])
    d = (ctrl - m) / ctrl * 100
    v = "reference" if a.startswith("control") else (
        f"**{d:.1f}% faster**" if d > noise else
        (f"{-d:.1f}% slower" if -d > noise else "indistinguishable"))
    print(f"| {a} | {m:.4f} s | {d:+.1f}% | {v} | "
          f"{'identical' if checks.get(a)==base_ck else '**differs**'} |")
json.dump({"noise": noise, "medians": results, "checksums": checks},
          open(f"{SC}/harness2.json", "w"), indent=1)
