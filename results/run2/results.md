# Sweep Results -- EE 451 Final Project

BLAS-backed refresh of the full `{1,2,4,8}`-thread sweep after swapping KNN and
MLP to vendor BLAS kernels. Job **3278100**, `shared` partition, node
`d17-03`, exit `0:0`. The merged CSVs in this folder were reconstructed from
[jobsweep.out](../../jobsweep.out); compiled binaries from the CARC run are in
[binaries/](binaries/). The archived pre-BLAS sweep remains in
[../run1/](../run1/).

---

## Executive Summary

1. **KNN is the big BLAS win.** Our pthreads KNN at `T=8` drops from
   **51,548 ms** in the archived run to **1,602 ms** here, a **32.2x**
   absolute improvement. The library comparison flips from **sklearn 16x
   faster** to **ours 2.2x faster**.
2. **MLP gets much faster in absolute time, but stops scaling.** Serial MLP
   drops from **42,009 ms** to **5,322 ms** (**7.9x faster**), but at `T=8`
   the pthreads wrapper is **6,335 ms**, only **0.84x** speedup versus the
   new serial BLAS baseline. The dense kernel got fixed; the thread-scaling
   story disappeared.
3. **SVM, DT, and NB look qualitatively the same as before.** SVM still scales
   well (**5.89x**), DT still scales worst (**1.92x**), NB still lands in the
   middle (**3.49x**), and pthreads still beats OpenMP on SVM / DT / NB.
4. **We now win every sklearn matchup at `T=8`.** SVM: **14.4x** faster. KNN:
   **2.2x** faster. MLP: **2.0x** faster. DT exact: **3.3x** faster. NB:
   **24.2x** faster. XGBoost histogram DT remains the only external baseline
   faster than our code.
5. **Accuracy is still a non-event.** KNN shifts only from `0.8104` to
   `0.8101`; MLP stays at `0.7748`; every algorithm remains within the same
   accuracy band as the archived run and its sklearn counterpart.

---

## Old Vs. New

Archived run: [../run1/results.md](../run1/results.md) (job `3272373`, pre-BLAS KNN /
MLP). Current run: this file (job `3278100`, BLAS-backed KNN / MLP).

### C++ side: serial and `T=8` pthreads

| Algo | Old serial ms | New serial ms | Serial delta | Old pthreads `T=8` ms | New pthreads `T=8` ms | `T=8` delta | Old `S_8` | New `S_8` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 333.00 | 334.36 | ~same | 55.34 | 56.78 | ~same | 6.02x | 5.89x |
| KNN | 323,019.07 | 9,436.16 | **34.2x faster** | 51,548.34 | 1,601.92 | **32.2x faster** | 6.26x | 5.89x |
| MLP | 42,008.91 | 5,322.25 | **7.9x faster** | 9,087.56 | 6,335.12 | **1.43x faster** | 4.62x | **0.84x** |
| DT  | 645.75 | 672.92 | ~same | 344.92 | 351.02 | ~same | 1.87x | 1.92x |
| NB  | 16.62 | 16.84 | ~same | 4.67 | 4.83 | ~same | 3.56x | 3.49x |

### Cross-framework at `T=8`

| Algo | Old verdict | New verdict |
|---|---|---|
| SVM | Ours **13.6x** faster than sklearn | Ours **14.4x** faster than sklearn |
| KNN | **sklearn 16.0x faster** | **Ours 2.2x faster** |
| MLP | Ours **1.4x** faster than sklearn | Ours **2.0x** faster than sklearn |
| DT  | XGBoost **1.9x** faster; ours **3.1x** faster than sklearn exact | XGBoost **1.86x** faster; ours **3.26x** faster than sklearn exact |
| NB  | Ours **24.5x** faster than sklearn | Ours **24.2x** faster than sklearn |

The two meaningful backend changes are completely concentrated in **KNN** and
**MLP**. SVM / DT / NB moved only within normal single-run noise on the same
node type.

---

## Hypothesis Vs. Reality

The proposal predicted:

- **MLP** would scale best (high arithmetic intensity, dense matmuls,
  SIMD-friendly).
- **KNN** and **NB** would hit memory-bandwidth limits early.
- **DT** and **SVM** would be synchronization- and convergence-limited.

What the refreshed BLAS-backed data says at `T=8` (pthreads variant, speedup vs
the `T=1` serial baseline from this run):

| Algorithm | Predicted rank | Actual rank | Speedup | Hypothesis |
|---|---|---|---|---|
| **KNN** | Memory-bound, early plateau | **1st (5.89x)** | Wrong |
| **SVM** | Sync-limited | 2nd (5.89x) | Wrong |
| **NB** | Memory-bound, early plateau | 3rd (3.49x) | Partially right |
| **DT** | Sync-limited | 4th (1.92x) | Mostly right |
| **MLP** | Best | **5th (0.84x)** | Completely wrong |

**Why the MLP prediction failed even harder after the rewrite.** The BLAS
backend fixes the dense math itself, but our network is still tiny
(`103 -> 64 -> 32 -> 1`) and `batch=128` produces very small GEMMs
(`128x103`, `128x64`, `128x32`). The serial BLAS path already captures almost
all the available kernel speed; wrapping it with outer OpenMP or pthreads adds
thread-management overhead without adding enough arithmetic to amortize it.

**Why the KNN prediction failed again.** Once we switched KNN to the standard
`||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b` SGEMM formulation, the old 16x
algorithmic deficit disappeared. The remaining work at `T=8` is mostly clean
query-block parallelism plus exact top-k selection, which scales well on 8
cores.

**Why the DT prediction stayed right.** Histogram CART still has a sequential
tree-growth critical path and per-node synchronization overhead. The BLAS
rewrite touched nothing in DT, so its shape is unchanged.

---

## C++ Sweep: Total_ms By Thread Count

Full data: [sweep_results_cpp.csv](sweep_results_cpp.csv).

### Serial baseline (`T=1` serial variant)

| Algo | train_ms | infer_ms | total_ms | accuracy |
|---|---:|---:|---:|---:|
| SVM | 330.74 | 3.62 | 334.36 | 0.7301 |
| KNN | 20.71 | 9,415.46 | 9,436.16 | 0.8101 |
| MLP | 5,306.71 | 15.54 | 5,322.25 | 0.7748 |
| DT  | 671.01 | 1.90 | 672.92 | 0.7627 |
| NB  | 12.34 | 4.51 | 16.84 | 0.6946 |

KNN is still inference-dominated, but no longer by three orders of magnitude.
The BLAS rewrite collapsed the serial constant from minutes to seconds.

### Speedup at `T=8` (vs `T=1` serial)

| Algo | OMP total_ms | OMP speedup | pthreads total_ms | pthreads speedup |
|---|---:|---:|---:|---:|
| SVM | 82.00 | **4.08x** | 56.78 | **5.89x** |
| KNN | 1,600.39 | **5.90x** | 1,601.92 | **5.89x** |
| MLP | 5,877.03 | **0.91x** | 6,335.12 | **0.84x** |
| DT  | 407.65 | **1.65x** | 351.02 | **1.92x** |
| NB  | 6.38 | **2.64x** | 4.83 | **3.49x** |

### OMP vs pthreads

pthreads still wins clearly on **SVM**, **DT**, and **NB**. KNN is a dead tie
at `T=8` (5.90x vs 5.89x). MLP is different now: the best variant is not a
parallel wrapper at all, but the serial BLAS path. OMP and pthreads both add
overhead around a kernel that is already near its local optimum.

### Accuracy parity (serial vs parallel)

All five algorithms still agree to 4 decimals across serial / OMP / pthreads.
Unlike the archived run, the refreshed BLAS-backed MLP also shows exact
accuracy parity at the printed precision (`0.7748` in all three variants).

---

## Cross-Framework Comparison At `T=8`

Full data: [sweep_results_sklearn.csv](sweep_results_sklearn.csv).

| Algo | Ours (pthreads) | sklearn | XGBoost | Runtime verdict | Accuracy verdict |
|---|---:|---:|---:|---|---|
| SVM | 56.78 ms | 817.52 ms | -- | **Ours 14.4x faster** | Tied (`73.01%` vs `73.15%`) |
| KNN | 1,601.92 ms | 3,527.64 ms | -- | **Ours 2.2x faster** | Tied (`81.01%` vs `81.06%`) |
| MLP | 6,335.12 ms | 12,815.04 ms | -- | **Ours 2.0x faster** | **Ours +2pt** (`77.48%` vs `75.46%`) |
| DT  | 351.02 ms | 1,145.80 ms (exact) | **188.67 ms (hist)** | XGBoost `1.86x` faster; ours `3.26x` faster than sklearn | Tied (`76.27% / 76.52% / 76.32%`) |
| NB  | 4.83 ms | 116.82 ms | -- | **Ours 24.2x faster** | Identical to sklearn hybrid (`69.46%`) |

### NB likelihood ablation (sklearn side only)

| Variant | Accuracy |
|---|---:|
| Hybrid (Gaussian + Bernoulli, matches ours) | 69.46% |
| Gaussian-only | 64.30% |
| Bernoulli-only | **70.24%** |

The NB side story is unchanged: the hybrid matches our implementation exactly,
and Bernoulli-only still edges it slightly on this test split.

---

## The Key Insight

The archived run already showed that **kernel formulation** dominated the KNN
comparison. This refreshed run confirms it more strongly: once we adopted the
same SGEMM-style reformulation on our side, the entire KNN verdict flipped.

At `T=1`:

- old ours: **323,019 ms**
- new ours: **9,436 ms**
- sklearn brute (`n_jobs=1`): **20,015 ms**

At `T=8`:

- old ours: **51,548 ms**
- new ours: **1,602 ms**
- sklearn brute (`n_jobs=8`): **3,528 ms**

The practical lesson is not "BLAS makes everything scale." It is:

1. **Fix the inner kernel first.** KNN went from a direct triple loop to an
   SGEMM-backed formulation and immediately got 30x-plus faster.
2. **Then ask who should own the parallelism.** For KNN, our outer
   query-block scheduling around single-threaded BLAS works well. For MLP, the
   serial BLAS kernel is already the right granularity and the outer wrappers
   just add overhead.
3. **For the non-BLAS algorithms, the old story still holds.** SVM and NB win
   because sklearn is serial by default; DT loses only to a mature histogram
   implementation (XGBoost), not to sklearn exact-split.

---

## Arithmetic Intensity Analysis

The original roofline-style prediction was `MLP >> DT >> {SVM, KNN, NB}`. The
refreshed run makes that ranking look even worse than the archived one.

### Predicted vs observed ranking at `T=8` (pthreads)

| Algorithm | Est. AI | AI-predicted rank | Observed speedup | Observed rank | What actually dominated |
|---|---:|:---:|---:|:---:|---|
| **KNN** | ~0.75 | 4th | **5.89x** | **1st** | SGEMM reformulation + embarrassingly parallel query blocks |
| **SVM** | ~1 | 3rd | 5.89x | 2nd | Only 20 synchronization events total |
| **NB** | ~0.75 | 4th | 3.49x | 3rd | Tiny absolute runtime + Amdahl floor |
| **DT** | ~3 | 2nd | 1.92x | 4th | Node-level critical path |
| **MLP** | ~42 | **1st** | **0.84x** | **5th** | Tiny GEMMs; BLAS already captures the dense kernel at `T=1` |

### Why the AI-only prediction is wrong here

1. **Kernel formulation changed the constants by orders of magnitude.** KNN is
   the clearest example: same algorithmic class, completely different runtime
   once the inner distance kernel is rewritten as SGEMM.
2. **Sync-event count still matters for the non-BLAS algorithms.** SVM, DT,
   and NB keep the same qualitative ordering as the archived run because their
   synchronization structure did not change.
3. **For BLAS-backed kernels, problem size matters more than raw AI.** MLP has
   the highest arithmetic intensity, but its minibatch matrices are too small
   to benefit from `T=8` BLAS threading on this node. The serial BLAS path wins
   because it avoids parallel overhead around tiny kernels.

### The correct predictor for the refreshed run

For this final codebase, the right question is no longer just "how much
FLOP-per-byte does the algorithm have?" It is:

```
1. Can the inner kernel be reformulated into a tuned primitive?
2. If yes, is the kernel large enough for threaded BLAS to pay off?
3. If not, how much useful work happens between synchronization events?
```

That three-part rule explains all five refreshed results:

- **KNN**: yes BLAS, yes enough outer parallel work -> excellent.
- **MLP**: yes BLAS, no large-enough kernels -> absolute win, scaling loss.
- **SVM / NB / DT**: no BLAS reformulation -> old sync-frequency story still applies.

---

## Open Questions / Future Work

1. **Does BLAS-backed KNN keep scaling past 8 threads?** The refreshed run gets
   us to `1.60 s` at `T=8`; the next question is where memory bandwidth or
   top-k selection becomes the limiting factor on `nlp` at `T=16,32,64`.
2. **How large does MLP need to be before BLAS threading pays off?** The
   obvious follow-up is `256 -> 128 -> 64` hidden sizes with the same training
   loop.
3. **Why are we now faster than sklearn on KNN by 2.2x?** Likely answer:
   lightweight C++ orchestration around the SGEMM panels plus exact top-k
   staying in-process. Worth a short profiling pass before claiming more than
   that.
4. **Can we close the DT-vs-XGBoost gap?** XGBoost still wins by `1.86x` on the
   histogram tree baseline.

---

## Files In This Folder

- [results.md](results.md) -- this file
- [../run1/](../run1/) -- archived pre-BLAS CARC sweep and figures
- [jobsweep.out](../jobsweep.out) -- full SLURM stdout from job `3278100`
- [sweep_results_cpp.csv](sweep_results_cpp.csv) -- merged C++ side, 60 rows
- [sweep_results_sklearn.csv](sweep_results_sklearn.csv) -- merged sklearn +
  XGBoost side, 32 rows
- [analytics_results_t{1,2,4,8}.csv](analytics_results_t1.csv) -- per-thread
  C++ intermediates reconstructed from the merged sweep
- [sklearn_results_t{1,2,4,8}.csv](sklearn_results_t1.csv) -- per-thread
  sklearn intermediates from the CARC BLAS run
- [binaries/](binaries/) -- compiled executables as built on `d17-03`.
  `-march=native` was used, so these binaries are microarchitecture-specific.
