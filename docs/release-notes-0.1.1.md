# Release Notes - 0.1.1

Date: 2026-02-24

## Highlights

- Full benchmark reliability pass is now trustworthy for release gating.
- MDX23C models run with MLX-native support, including optional compiled forward acceleration.
- Loader hardening resolves major VR/MDX/MDXC mismatch classes seen in prior full-catalog runs.
- Benchmark diagnostics now retain actionable exception context instead of only surface symptoms.

## Runtime Policy

- `experimental_compile_model_forward` remains an opt-in acceleration path.
- Compiled-forward acceleration is currently targeted at MDX23C in MDXC.
- Roformer compile paths (`experimental_compile_shapeless`, `experimental_roformer_static_compiled_demix`) are accepted for compatibility but currently ignored by policy.

## Release Evidence

- Unit suite green on release candidate branch.
- Full benchmark gate passes with zero failures on the release benchmark corpus/settings.
