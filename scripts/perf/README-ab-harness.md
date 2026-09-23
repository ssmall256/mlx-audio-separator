# ABBA A/B harness

The only tool in this repo that can tell a 3% change from thermal drift.

```bash
AB_CLIP=/path/to/clip.npy DEMUCS_CACHE=~/.cache/demucs-mlx \
AB_PYTHON=.venv/bin/python \
  python scripts/perf/ab_harness.py 60 7 4      # seconds, reps, rounds
```

It prints a table with a **noise floor** measured from two same-config control
arms. Anything smaller than that floor has not been shown to do anything.

## Why it is shaped this way

Three properties, each of which exists because its absence produced a wrong
answer in this repo:

- **One process per arm.** Reassigning `type(obj).__call__`, or mixing arms
  whose tensor shapes differ, thrashes MLX's compile cache and inflates every
  arm together. It is silent: the run stays internally consistent and only
  disagrees with a clean baseline. A 0.435 s baseline became 2.19 s this way.
- **Code-change arms are source patches applied before import**, so no arm can
  leak state into another.
- **Two same-config control arms.** `docs/roformer-kernel-fusion-followup.md`
  records two Demucs "wins" of -16% and -20% that had to be discarded because a
  control moved. The floor is now computed automatically rather than being
  something to remember.

Rounds rotate the arm order, because a fixed ABBA order samples only two
positions and the first arm in a round pays a warm-up penalty.

## It will reject a bad machine, and should be allowed to

Run on a loaded laptop this harness reports a noise floor around 40% and five
"wins" of 46-56%. The per-round trace shows every arm in round 1 at 0.65-0.87 s
and every arm in round 3 at 1.94-2.90 s -- a monotonic 3x thermal slide that no
ordering scheme survives. On an idle M4 the same run gives a 0.36-0.62% floor.
If the floor comes back above a few percent, the measurement is void; move to a
quiet machine rather than reading the table.

## Adding an arm

Env-var arms go in the block near the top of `ab_arm_runner.py`. Arms needing a
code change go in `PATCHES` as `(relative path, exact old text, new text)`.
Arms whose output should be unchanged must show `identical` in the output
column; a ceiling arm that deliberately breaks numerics will show `differs` and
is only good for bounding what a real fix could win.
