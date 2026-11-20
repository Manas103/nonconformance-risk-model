# Nonconformance Risk Model on Production-Line Records

A LightGBM model that scores a manufactured part's risk of failing final
inspection from its station path and per-station measurement columns,
beaten against a logistic-regression baseline and a random-inspection
floor, calibrated with isotonic regression, and tested on the one slice
that actually matters for a new production line: parts whose station path
never appeared in training. Every number below was measured on this
machine by running the code in this repository, not targeted in advance.

## Why this exists

"End-to-end ML pipelines on real, messy datasets" needs a dataset that is
actually messy: extreme class imbalance, hundreds of mostly-missing
measurement columns, and a many-to-many relationship between parts and
the stations they pass through. A nonconformance risk model is only
useful if it is honest about where it stops working, which is why the
cold-start slice (unseen station paths) is reported as its own number,
not folded into the headline metric.

## What this is and is not

- **Not the real Kaggle Bosch Production Line Performance dataset.** That
  dataset requires an authenticated Kaggle account that has accepted the
  competition rules; no `~/.kaggle/kaggle.json` and no `KAGGLE_USERNAME`/
  `KAGGLE_KEY` existed in this build environment, so it could not be
  downloaded. No legitimate public mirror was found either: a Zenodo
  record and a Hugging Face dataset both named "bosch" turned out to be
  an unrelated software package and an unrelated car-manual QA corpus
  respectively, confirmed by fetching their actual contents, not assumed
  from the name. `risk_model/generate_data.py` instead generates a
  synthetic dataset matching the real one's structural character: 48
  stations, 16 measurement columns per station (771 total, close to the
  real 968), a part visiting a random subset of stations with a
  Zipf-weighted popularity (producing the same long-tailed
  station-path-combination structure the real dataset is known for), and
  a failure rate calibrated toward the real dataset's 0.58%. It is not
  real Bosch records, and no claim on this page implies it is.
- **Missingness is structural, not imputed-then-hidden.** A part's
  measurement columns are null for every station it did not visit;
  LightGBM splits on missing values natively, so there is no imputation
  step that could leak information. The logistic-regression baseline, by
  contrast, needs imputation (zero-fill) because it cannot handle nulls,
  which is itself part of what makes it a baseline rather than the model.
- **The risky station combinations and marker measurement thresholds
  that drive the synthetic failure label are not given to the model as
  features.** Only generic structure (one-hot station-visited flags,
  visit count, missingness fraction, the raw measurement columns
  themselves) is exposed; the model has to find the risky combinations
  through tree splits, the way a real model would.
- **Cold-start means the exact station-path combination, not an unseen
  individual station.** With 48 stations and 4-14 visited per part, most
  station-path combinations in a 1.18M-row set are rare or unique, so
  81.3% of the test split's rows qualify.
- **Machine and toolchain.** 8 physical / 16 logical cores, Windows 11
  Home, Python 3.12.10, LightGBM 4.7.0, scikit-learn 1.9.1, DuckDB 2.0.0,
  pandas 3.1.1. CPU only.

## Architecture

```
risk_model/
  generate_data.py     synthetic production-line generator: Zipf-weighted station paths, per-station
                         measurement columns null for unvisited stations, a log-odds failure model driven
                         by 3 risky station-set combinations and 4 marker measurement columns
  features.py              DuckDB SQL: one-hot station-visited flags, visit count, missingness fraction;
                            a time-ordered (by Id) train/val/test split; a cold-start mask (station_path
                            unseen in train)
  train.py                   LightGBM + logistic-regression + random-inspection baselines; a
                              best-threshold-by-MCC search (0.5 is meaningless at a 0.5%-ish positive
                              rate); a 5%-inspection-budget recall; isotonic calibration; the cold-start
                              evaluation
scripts/
  run_benchmarks.py             builds features, trains everything, writes docs/benchmark_output.json
tests/
  test_generate_data.py            determinism, null-for-unvisited-station structure, binary response,
                                     low-but-nonzero failure rate at scale
  test_features.py                   station-flag correctness against the raw station_path string,
                                       time-ordered split integrity, cold-start mask correctness
  test_train.py                        best-threshold-MCC on a separable vs. a random synthetic check,
                                        inspection-budget recall edge cases, calibration-error sanity checks
docs/
  test_output.txt, benchmark_output.json
```

### Why the classification threshold is chosen by MCC, not fixed at 0.5

At a 0.55% positive rate, a threshold of 0.5 almost never fires: nearly
every prediction is correct by predicting "no failure" for everything,
which is a useless, high-accuracy, zero-MCC model. `best_threshold_mcc`
scans 99 thresholds on the validation split and picks the one that
maximizes MCC there, then applies that fixed threshold to the test split,
so the threshold itself is never chosen by looking at test labels.

## Validation

```
$ pytest tests -v
18 passed in 17.30s
```

Full output: `docs/test_output.txt`. Layers: generator determinism and
structural invariants (a station's measurement columns are null if and
only if that station is absent from `station_path`); the DuckDB feature
builder's one-hot flags checked row by row against the same
`station_path` string they were built from; the time-ordered split's
non-overlap and the cold-start mask's set-membership logic; every
training-metric function (threshold search, inspection-budget recall,
calibration error) checked against hand-constructed cases with a known
right answer, independent of the real training run.

## Findings

**The cold-start slice does not collapse the way the claim says it
should.** The claim is that MCC falls to 0.11 on unseen station paths.
Measured: cold-start MCC is 0.136, slightly *higher* than the overall
0.129, not lower. Root cause, read from the generator: the failure
signal here comes from two independent sources, 3 risky *exact* station
combinations and 4 marker *measurement* thresholds that fire regardless
of which stations the part visited. A part with an unseen station-path
combination can still trip a marker-measurement threshold, and with 971
of 1,180,000 parts failing overall, the marker-driven signal dominates
the combination-driven signal enough that losing the exact-path memory
costs the model almost nothing. This is reported as measured, not
adjusted to match the claim; a dataset whose failure signal depended
*only* on station-path combinations would show the claimed collapse.

**Logistic regression beats LightGBM by a hair.** 0.1320 MCC versus
0.1288. At this feature count (818, mostly indicator-like) and this
signal structure (threshold-based marker rules, which a sufficiently
deep tree should find at least as easily as a linear model), that is a
close enough margin to be within the run's own noise rather than a real
LightGBM weakness; it is reported because the point of a baseline is to
show when the fancier model does not obviously win, not to manufacture a
gap.

## Measured results

8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, CPU only.
826,000 train / 177,000 val / 177,000 test rows (time-ordered split of
1,180,000 synthetic parts), 818 features, LightGBM training time 87.8s.

| metric | claim | measured | meets claim |
|---|---|---|---|
| Dataset provenance | public Bosch dataset, 1.18M real parts, 968 columns, 0.58% failures | synthetic substitute, 1.18M parts, 771 columns, 0.55% failures (see "What this is and is not") | no, disclosed substitution |
| **LightGBM test MCC** | 0.41 | **0.1288** | no |
| Logistic regression baseline MCC | 0.20 | 0.1320 | no (and edges out LightGBM, see Findings) |
| Random inspection baseline MCC | 0.00 | -0.0036 | yes |
| Failures caught at 5% inspection budget | 46% | 42.6% | no, close |
| Expected calibration error after isotonic calibration | 0.012 | 0.000185 | yes, better than claimed (uncalibrated: 0.100) |
| MCC on cold-start (unseen station path) slice | falls to 0.11 | 0.1361, does not fall | no, see Findings |

**What each number does and does not measure.** MCC (not accuracy, not
F1) is reported throughout because accuracy is meaningless at a 0.55%
positive rate (predicting "no failure" always scores ~99.45% accuracy)
and MCC is the one scalar metric that stays informative at extreme
imbalance. The cold-start slice is 81.3% of the test split (143,979 of
177,000 rows) by construction, since most exact station-path combinations
in a 1.18M-row set with 48 stations and 4-14 visited per part are rare;
this is a property of the generator's combinatorics, not a tuned split
fraction.

## Building and running

```
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe risk_model/generate_data.py --n-parts 1180000   # ~6-10 min, writes data/production_records.parquet
venv/Scripts/python.exe scripts/run_benchmarks.py                       # ~3.5 min: features, train, calibrate, cold-start
venv/Scripts/python.exe -m pytest tests -v
```

## Limitations

- The dataset is entirely synthetic (see "What this is and is not"); none
  of these numbers say anything about a real production line.
- The synthetic failure model mixes a combination-driven signal and a
  measurement-threshold-driven signal in a way that happens to make the
  cold-start slice easy; a dataset whose failure signal depended more
  heavily on station-path combinations specifically would likely show a
  real cold-start collapse, and this generator cannot be used to measure
  that.
- The risky-station-set and marker-measurement-column lift values were
  chosen by hand to land the overall failure rate near 0.58%, not fit to
  any real distribution; the specific LightGBM MCC of 0.1288 is a
  property of this generator's particular signal strength, not a general
  claim about how hard real nonconformance prediction is.
