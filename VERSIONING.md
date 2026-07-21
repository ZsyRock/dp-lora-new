# Version and GitHub workflow

## Branches

- `main` is the integrated branch. It should stay runnable and pass the test
  workflow.
- `codex/<topic>` branches isolate one coherent change. They can be rewritten or
  discarded without destabilising `main`.
- Changes reach `main` through a pull request. New research implementations start
  as Draft PRs until tests and the intended experiment contract are reviewed.

## Commits, pull requests and releases

A commit is an immutable snapshot identified by a SHA. Keep each commit focused
and record that SHA with every HPC run. A pull request compares a branch with
`main`; it is the review and CI gate, not a released version.

This repository uses squash merging for focused feature branches. That leaves
one explanatory commit on `main`. After the merge, the accepted `main` commit is
the reproducibility anchor.

A tag gives one accepted commit a permanent human-readable name. A GitHub Release
adds release notes to that tag. Tags are never moved after publication.

## Version numbers

The project follows Semantic Versioning during the `0.x` research phase:

- `0.MINOR.0` changes the supported method or experiment contract;
- `0.MINOR.PATCH` fixes behaviour without changing the contract;
- `rcN` marks a release candidate that still needs integration or HPC evidence.

For example, Python uses `0.2.0rc1`, while the matching GitHub tag is
`v0.2.0-rc.1`. Promote it to `v0.2.0` only after CI, the GPU pilot and review all
pass.

## Experiment provenance

Every published result should retain the Git commit SHA, command/configuration,
random seeds, dataset/model revisions, Python and package versions, GPU/driver,
and Slurm job ID. Large checkpoints, raw results and exact private diagnostics do
not belong in Git. Store them in controlled experiment storage; exact diagnostic
files must not be released as DP outputs.
