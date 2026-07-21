# Changelog

This project follows Semantic Versioning while it is in the `0.x` research
phase. GitHub tags use `v0.2.0-rc.1`; Python package metadata uses the equivalent
PEP 440 form `0.2.0rc1`.

## 0.2.0rc1 - 2026-07-21

- Define the comparison as fixed-C DP-LoRA versus the same baseline with the
  main SlaClip controller; remove SlaClip-Q from the experiment surface.
- Apply flat record-level clipping to every trainable LoRA and classification
  head parameter and fail closed on unsupported or uncovered parameters.
- Correct mean-loss per-sample gradient scaling, Poisson sampling/accounting,
  virtual-batch updates and resumable DP state.
- Add exact, opt-in private gradient diagnostics with an explicit non-DP
  acknowledgement, per-parameter summaries, logical-step traces and Git-ignored
  output paths.
- Add paired SST-2 runners, five classification metrics and reproducible sweep
  entry points.
- Handle empty Poisson draws as real noise/accounting steps and isolate SlaClip's
  auxiliary RNG so paired runs retain the same gradient-noise stream.
- Add unit tests, GitHub Actions, tested dependency constraints and an HPC Slurm
  template.

This is a release candidate. A stable `v0.2.0` tag should only be created after
the Draft PR passes CI and a real GPU pilot completes on the target HPC system.
