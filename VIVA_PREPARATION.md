# ExoScope — Viva & Presentation Preparation

Everything here matches the code in this folder. File references let you show the examiner the exact
implementation.

## 1. The 60-second explanation

> When a planet crosses its star, the star dims slightly, and the dip repeats every orbit. ExoScope downloads
> real brightness measurements from NASA's Kepler and TESS missions, cleans them, and searches for repeating dips
> with two methods: a threshold detector that finds individual dips, and Box Least Squares, which adds up every
> orbit at many trial periods. The Discrete Mathematics is part of the pipeline itself. The threshold detector's
> period is an approximate GCD of the gaps between dips. The two methods are compared as sets. Events form a graph
> whose edges mean "a whole number of orbits apart". Combinatorics drives systematic experiments over the settings.
> On nine known planets, the BLS period matched NASA's archive within 0.22%. The app always reports *candidates*,
> because eclipsing binary stars can mimic planets, and it includes checks that catch them.

## 2. The 2–3 minute technical explanation

1. **Data** (`exoscope/data/retrieval.py`). Lightkurve searches the MAST archive. We keep the pipeline-corrected
   PDCSAP flux, time and the mission's quality flags. A two-level cache (FITS files plus our own compact CSV and
   JSON) makes repeated use offline-capable. Three real light curves are bundled.
2. **Preprocessing** (`exoscope/processing.py`). The steps are: drop NaNs, apply the quality mask, normalise by
   the median, then detrend with a Savitzky–Golay filter fitted per data segment. That filter uses iterative
   sigma-masking, so transits don't pull the trend down. Finally we clip only *upward* outliers (clipping downward
   would delete transits) and optionally bin. Every step is logged with point counts.
3. **Threshold detector** (`detection/threshold.py`). Noise σ = 1.4826·MAD, which is robust to the transits
   themselves. Points below median − kσ are flagged, consecutive flagged points are grouped into events, and runs
   shorter than *m* points are rejected. The period comes from the **approximate GCD** in
   `dmath/number_theory.py:estimate_period`.
4. **BLS** (`detection/bls.py`). Astropy `BoxLeastSquares` runs on a uniform frequency grid with
   Δf = D_min / T², capped at 30,000 periods, over several trial durations. It reports period, t₀, duration,
   depth, SNR and SDE. The signal is then expanded into individual predicted epochs so it can be compared with
   the threshold events.
5. **Comparison** (`dmath/sets.py`). Events within ±τ hours are matched one-to-one by the Hungarian algorithm.
   This gives U, A, B, the gap set G, all the set operations, and the Jaccard index.
6. **Evaluation** (`evaluation.py`). A 10-point checklist: repeated events, intervals being integer multiples of
   P, SNR, odd/even depths, box versus sinusoid, secondary eclipse, data coverage, algorithm agreement, depth and
   duration. The verdict is *gated*: SNR < 7.1 means "no significant signal", and any failed false-positive test
   means "non-planetary signs".
7. **Lab, storage, reports.** Five Discrete Maths pages, SQLite history, and CSV/JSON/Markdown/HTML exports.

## 3. Problem, solution, contribution

- **Problem:** find periodic transit signals in noisy light curves, compare methods fairly, and avoid
  over-claiming planets.
- **Solution:** a reproducible pipeline (retrieve → preprocess → detect twice → compare → evaluate) with an
  interactive interface.
- **Contribution (novelty):** the Discrete Mathematics does real work. Number theory produces a period estimate
  that survives missed and false dips, which the original median-spacing method could not. Set theory gives an
  exact, explainable comparison of the detectors. Graph theory exposes which dips support each other.
  Combinatorics sizes and runs experiments. Each module is visual and tested.

## 4. Architecture and data flow

```
MAST (Lightkurve) / bundled CSV / synthetic generator
          │  LightCurveData (time, flux, flux_err, quality, meta, truth?)
          ▼
   preprocess()  ──►  ProcessedLightCurve (+ step log)
          │
   ┌──────┴────────┐
detect_threshold   run_bls  ──►  per-epoch events B, gap epochs G
(events A,               │
 approx-GCD period)      │
   └──────┬────────┘
          ▼
compare_detections (sets U, A, B)  ──►  evaluate_candidate (checklist, verdict)
          │
AnalysisResult ──► Streamlit pages · SQLite history · reports
                └─► dmath modules (sets, number theory, graphs, relations, combinatorics)
```

The UI (`views/`, `exoscope/ui/`) is separate from the logic (`exoscope/`), so the logic is unit-tested without
the browser.

## 5. The important algorithms and design decisions

**Approximate GCD (`estimate_period`).**
1. Candidate periods are Δᵢⱼ / k for pairs of events and k = 1…10. Dividing by k covers k − 1 missed transits.
2. For each candidate P, find the largest set of events that are congruent modulo P within ε: the circular
   distance of (tᵢ − tⱼ) mod P is ≤ ε.
3. Score = members × fill fraction, where fill = members / epochs spanned. Without the fill term, P/2 or P/3 ties
   with P, and can even win by absorbing a stray dip. That bug happened during development and is now covered by
   a test.
4. Assign integer epochs nᵢ and fit tₙ = t₀ + nP by least squares. The residuals are the O−C diagram.

**Why tolerances everywhere?** Real times are noisy, so "divides" and "congruent" become "within ε". As a
consequence, exact integer GCDs fail (e.g. gcd(24, 23, 48) = 1), and the consistency relation is not transitive.

**Why match before comparing sets?** The two methods never report identical timestamps, so equal elements must
be defined first. BLS is expanded into epochs so both sets contain the same type of object.

**Why only upward clipping?** Transits are downward points; clipping them would remove the signal.

**Why the gated verdict?** BLS tries about 10⁵ period/duration combinations, so the best peak in pure noise
reaches SNR 4–6 (the look-elsewhere effect). The first evaluation run showed 5/10 noise curves reported as
"possible". Treating SNR < 7.1 as not detected gave 0/30.

**Why inflate the per-point errors?** Injection–recovery revealed that the archived Kepler/TESS errors are
1.3–1.9× smaller than the measured scatter (residual stellar variability and red noise). Because BLS SNR divides by those errors,
real-data SNRs were optimistic. `run_bls` now scales errors up to the robust scatter (never down) and reports the
factor. Kepler-10's SNR went from 67 to 43.5, which is still far above 7.1.

**Why a rule checklist rather than machine learning?** Every decision can be explained and traced to a number.
The CNN approach (Shallue & Vanderburg 2018) needs labelled training data and is less explainable.

## 6. Likely viva questions (with answers)

1. **What is a transit and what does its depth tell you?** A planet blocking starlight. δ ≈ (Rp/Rs)², so depth
   gives the radius *ratio*. For Kepler-10, 134 ppm → Rp/Rs ≈ 0.012 → about 1.3 Earth radii with Rs = 1.06 R☉
   (the archive value is 1.47 R⊕; our box-model depth is shallower).
2. **Why two detection methods?** They have different strengths. The threshold method is fast and explainable
   per dip, but it needs each transit above the noise. BLS combines all transits, so it finds shallow signals.
   Kepler-10 shows it: 39 threshold events vs 102 BLS epochs.
3. **Define the universe U in your set comparison.** U = A ∪ B ∪ G, where G = BLS-predicted epochs falling in
   data gaps. That makes (A ∪ B)ᶜ = G meaningful: expected but unobservable transits.
4. **What does the Jaccard index mean here?** |A∩B| / |A∪B|. It's 1.00 for Kepler-8 (deep transits, full
   agreement) and 0.33 for Kepler-10 (BLS sees many transits the threshold can't).
5. **How do you find the period from event times?** Approximate GCD via the largest congruence class, then a
   least-squares fit. See section 5.
6. **Why not just take the median gap?** One missed transit makes a gap of 2P, and flagging single points makes
   gaps equal to the cadence. On Kepler-8, the old method gives 0.0204 d (the cadence); ours gives 3.5221 d
   (archive 3.5225 d).
7. **What is a congruence class here?** The events t with (t − t₀) mod P ≈ 0. They all fall at the same angle on
   the modular clock.
8. **Define your graph's edges.** {u, v} ∈ E iff | |t_v − t_u| − nP | ≤ ε for some integer 1 ≤ n ≤ N, optionally
   with similar depths.
9. **Why is the consistency relation not an equivalence relation?** It is reflexive and symmetric but not
   transitive. Small timing errors add up along a chain, and pairs more than N periods apart get no edge.
   Components equal the classes of its transitive closure.
10. **Component vs clique?** A component is linked by chains; a clique has every pair consistent, which is
    stronger evidence. Kepler-10 has a largest clique of 7 among 39 vertices.
11. **Is "event → algorithm" a function?** No. It is many-to-many, because an event can be found by both. A
    function needs exactly one output per input.
12. **Where does a function check catch a real problem?** Observation time → flux must be a function, so
    duplicate timestamps are removed. Event → epoch must be injective, otherwise one transit was split into two
    detections.
13. **How many configurations does your experiment have?** By the rule of product, 4 thresholds × 3 minimum
    points × 2 windows = 24. A constraint (flatten window ≥ 3 × longest duration) leaves 12. Only one distinct
    preprocessing run is needed: the size of the projection.
14. **What is SDE vs SNR?** SNR = depth / depth error for the best model. SDE = how far the periodogram peak stands
    above the average peak, in standard deviations.
15. **How do you catch eclipsing binaries?** With the odd/even depth comparison (errors from epoch-to-epoch
    scatter) and a secondary-eclipse test at phase 0.5. On 30 simulated binaries, 30 were flagged.
16. **How accurate is it?** On 9 known planets, the BLS period error vs the archive was 0.001–0.22%. On 120
    synthetic light curves, BLS recovered 74% of periods within 1%, and 100% when per-point SNR ≥ 3. On pure
    noise, there were 0 false detections in 30. See `docs/EVALUATION.md`, generated by `scripts/evaluate.py`.
17. **What is injection–recovery and what did you find?** Add fake transits to real data and count how many are
    found. Each star's own planet is masked first. In the Kepler-10 light curve, 100 ppm transits were recovered
    58% of the time in real data vs 92% in white noise with the same scatter. 200 ppm and deeper were always
    recovered. For the noisier Kepler-8, the 50% limit is about 200 ppm. Real noise costs about 1.07× in SNR
    (`docs/EVALUATION.md` §4).
18. **Why Streamlit, why SQLite?** Streamlit gives an interactive Python UI without a separate frontend. SQLite
    is a single local file that needs no server, so it suits a reproducible coursework project.
19. **What's your SDG link?** Primary: SDG 9.5 (scientific research capability): open, reproducible methods on
    public mission data. Secondary: SDG 4 (education): interactive lessons. These are alignment statements, not
    claims of impact.

## 7. Difficult questions — answer honestly

- **"Did you discover a planet?"** No. Every signal is a *candidate*. The targets we validate against are already
  confirmed in the NASA archive; that confirmation comes from other observations, not from ExoScope.
- **"Your synthetic results look too good."** They use white noise. Real stars have red noise and systematics.
  That's why we also report injection–recovery on real light curves, plus real-planet results.
- **"Why a box model?"** BLS is the standard, fast baseline. Transit Least Squares (Hippke & Heller 2019) uses a
  realistic shape and is more sensitive to small planets. It's future work.
- **"Your checklist thresholds (7.1σ, 3σ, 3%) seem arbitrary."** They are documented conventions (7.1σ comes
  from the Kepler pipeline), not calibrated probabilities, and the app says so on the Results page.
- **"What about planets with fewer than 3 transits?"** One quarter is about 90 days, so long-period planets
  (e.g. Kepler-10 c, 45 d) can't be shown to be periodic. Stitching quarters together is future work.
- **"BLS's period is slightly less precise than your threshold method on deep transits — why?"** BLS evaluates
  a finite grid (capped at 30,000 periods). The approximate-GCD method fits the epochs by least squares. For
  shallow signals only BLS works.
- **"What if MAST is down?"** Use the bundled offline copies of real data. Cached downloads also work offline.
- **"Isn't the original code missing?"** The original repository wasn't available, so v1 was reconstructed
  from the project description (`exoscope/legacy.py`) and kept for a direct comparison.

## 8. If something is incomplete or fails, say this

- "That part is a documented limitation: for example, TLS and FFT searches are not implemented, and the box model
  makes depths approximate. Here is what *is* measured…" Then point to `docs/EVALUATION.md`.
- If a live download is slow: "MAST can be slow; this is the bundled offline copy of the same archive data."
- Never claim a number you haven't seen on screen or in `docs/EVALUATION.md`.
