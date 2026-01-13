# FDHC Cuckoo Hashing – Experimental Artifact

This repository accompanies the paper  
**“Fully Dynamic High-Load Cuckoo Hashing with Bounded Layer Growth”**.

The paper proposes a fully dynamic, multi-layer cuckoo hashing scheme that operates
reliably at extremely high load factors while maintaining predictable performance
and a bounded number of active layers. The key idea is to decouple structural growth
from conservative load thresholds and instead drive expansion through observed
insertion failure, combined with incremental promotion and controlled layer removal.
This enables near-optimal space utilization without global rehashing and without
unbounded structural depth.

This repository contains the full implementation of the proposed structure, a set of
comparison models, and all experimental scripts used to generate the results reported
in the paper.

---

## Implemented Structure

The proposed data structure is implemented in the `fdhccuckoo` module.  
It supports:
- Geometrically sized layers with a bounded active window
- Incremental promotion and safe layer removal
- Explicit tracking of placement indices
- Fully dynamic updates (insertions and deletions)
- Bounded expected insertion and lookup cost
- Detailed runtime and placement statistics for evaluation

The hash table may also be used purely as an indexing layer, storing compact references
or encoded representations instead of raw payloads, as discussed in the paper.

---

## Comparison Models

To evaluate the proposed scheme, we compare it against several representative hashing
approaches commonly used or studied in high-load settings:

- **Classical Cuckoo Hashing**  
  The original two-choice cuckoo hashing scheme, serving as a baseline for insertion
  failures and relocation behavior near saturation.

- **d-ary Cuckoo Hashing**  
  An extension of classical cuckoo hashing that increases the number of candidate
  locations per key. This improves load tolerance but relies on uninformed relocation
  and does not address dynamic expansion.

- **Bubble-Up Cuckoo Hashing**  
  A recent high-load scheme that prioritizes relocations to reduce insertion failures
  and achieve near-optimal space utilization in centralized tables. We use it as a
  reference point for high-load insertion cost, noting that it does not support
  incremental layering or distributed settings.

- **Dynamic / Incremental Variants**  
  Where applicable, we include models inspired by dynamic cuckoo constructions that
  avoid full rehashing but allow unbounded structural growth.

Not all competing schemes provide public reference implementations. In such cases,
we implemented faithful models based on the algorithmic descriptions in the original
papers, using consistent hashing primitives and evaluation metrics.

---

## Experiments

All experiments reported in the paper are reproducible using the scripts in this
repository. The experimental framework is divided conceptually into two parts:

- **Runners**  
  Scripts that execute simulations, insert keys until saturation or failure, and
  collect statistics such as probe counts, placement indices, per-layer occupancy,
  and insertion success rates.

- **Plots**  
  Scripts that generate publication-quality figures from the collected statistics,
  including CDFs of placement indices, per-layer utilization, insertion cost under
  high load, and sensitivity to configuration parameters (e.g., active-layer threshold).

We evaluate multiple table sizes and parameter settings, and report averaged behavior
over repeated runs with independent random seeds.

---

## Running the Code

A standard single-table simulation can be run directly from Python:

```bash
python -m fdhccuckoo.run_single_simulation

Experiments and plots are executed as modules from the repository root, for example:

```bash
python -m experiments.runners.run_high_load
python -m experiments.plots.plot_cdf