# Fully Dynamic Hop Countered (FDHC) Cuckoo Hashing

This repository accompanies the paper  
**“Fully Dynamic Hop Countered Cuckoo Hashing\\ for Efficient Secure Web3 Directory and Beyond”**.

The paper introduces a fully dynamic multi-layer cuckoo hashing scheme designed to operate reliably at extremely high load factors while maintaining predictable insertion costs and a bounded number of active layers.
Unlike traditional approaches that rely on conservative load thresholds or global rehashing, the proposed model decouples structural growth from fixed occupancy limits. Instead, table expansion is triggered only by observed insertion failures, allowing the structure to adapt precisely to actual load conditions.
The scheme combines incremental promotion, informed eviction decisions, and controlled layer removal to prevent unbounded structural growth. As a result, the structure achieves near-optimal space utilization without global rehashing and without sacrificing stability.
Beyond space efficiency, a key advantage of the proposed model is its significantly lower insertion cost under high load. Even in load regimes exceeding 95–99%, the scheme maintains a small and predictable number of probes per insertion, outperforming existing cuckoo-based baselines that experience sharp increases in relocation cost as load approaches capacity.
This repository contains the full implementation of the proposed model, along with multiple comparison baselines and all experimental scripts used to generate the results reported in the paper.

---

## Repository Structure

This repository contains the implementation of a fully dynamic multi-layer cuckoo
hashing scheme, together with baseline models and all experimental scripts used to
evaluate behavior under high load.

- **FDHCcuckoo/**  
  Implementation of the proposed FDIC data structure, including multi-layer management,
  insertion and relocation logic, and detailed runtime statistics.

- **Comparison_structures/**  
  Reference implementations of baseline schemes used for comparison, including
  classical cuckoo hashing, d-ary random-walk cuckoo hashing, stash-based variants,
  and bubble-up cuckoo hashing.

At the repository root, we provide dedicated entry-point scripts (`run_<model>.py`)
that execute a complete experiment for a specific model and print human-readable
summaries. These scripts are intended for direct execution and inspection.

---

## Experiments

The `Experiments/` directory contains the experimental framework used in the paper and
is organized into two subdirectories:

- **Experiments/runners/**  
  Simulation scripts whose sole purpose is to collect structured statistics for
  plotting. These scripts do not produce user-facing output and are used only to
  generate the data consumed by the plotting utilities.

- **Experiments/plots/**  
  Scripts that generate publication-quality figures from the statistics produced by
  the runners, enabling direct comparison of insertion cost, load tolerance, and
  high-load behavior across models.

---

## Comparison Models

To evaluate the proposed FDIC scheme, we compare it against several representative
cuckoo-hashing–based approaches that capture common design trade-offs in high-load
settings:

- **Classical Cuckoo Hashing (2-Choice)**  
  The original two-choice cuckoo hashing scheme, serving as a baseline for insertion
  failures and relocation behavior near saturation. While simple and well understood,
  it exhibits rapidly increasing insertion cost and early failures as the load
  approaches capacity.

- **d-ary Cuckoo Hashing with Random Walk**  
  An extension of classical cuckoo hashing that allows each key to choose among multiple
  candidate locations. Insertions are resolved using an uninformed random-walk
  relocation process, which improves load tolerance but incurs high insertion cost
  under heavy load and does not support dynamic growth.

- **d-ary Cuckoo Hashing with Stash**  
  A variant of the random-walk approach augmented with a small stash to temporarily
  store displaced keys. The stash delays insertion failures and increases achievable
  load, but does not fundamentally reduce relocation cost or provide controlled
  structural expansion.

- **Bubble-Up Cuckoo Hashing**  
  A recent high-load scheme that prioritizes relocations based on hash order to reduce
  insertion failures and achieve near-optimal space utilization in a single centralized
  table. We use it as a strong reference point for high-load insertion cost, noting that
  it does not support incremental layering or bounded structural depth.

- **FDIC (Proposed Model)**  
  Our proposed Fully Dynamic Incremental Cuckoo (FDIC) scheme combines multi-layer
  expansion, informed relocation, and controlled layer removal. Structural growth is
  driven by observed insertion failure rather than conservative load thresholds,
  enabling near-optimal space utilization while maintaining consistently low insertion
  cost even at extreme load factors.

Not all competing schemes provide public reference implementations. In such cases,
we implemented faithful models based on the algorithmic descriptions in the original
papers, using consistent hashing primitives, load definitions, and evaluation metrics
to ensure a fair comparison.

---

## Running the Code

A standard single-table simulation can be run directly from Python:

```bash
python -m run_FDHCcuckoo
python -m run_basicBubbleUp
```

Experiments and plots are executed as modules from the repository root, for example:

```bash
python -m experiments.plots.plot_compareResults
python -m experiments.plots.plot_occupancyPerLayer
```