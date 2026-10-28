# ExoScope: presentation notes

## One-line pitch
**An interactive exoplanet transit analysis platform that demonstrates how Discrete Mathematics is applied to
detect, compare, visualise and evaluate periodic astronomical events, using real NASA Kepler and TESS data.**

## Novelty (about 60 seconds)
Most transit tools plot data and report a period. In ExoScope the Discrete Mathematics **does work in the
pipeline**:

1. **Number theory finds the period.** The threshold detector's period is an *approximate GCD*: the largest set of
   events congruent modulo P. That makes it robust to missed transits (intervals of 2P, 3P …) and to false dips,
   where the original "median spacing" method failed.
2. **Set theory compares the algorithms.** Threshold dips and BLS epochs become sets A and B over a defined
   universe. A ∩ B is agreement, B − A is the shallow transits only BLS can see, and (A ∪ B)ᶜ is transits hidden in
   data gaps. Precision and recall are set operations.
3. **Graph theory shows which events support each other.** Events are vertices and "a whole number of orbits
   apart" is the edge rule. Components and cliques group consistent events, and isolated vertices flag noise.
   The tolerance relation is shown *not* to be transitive.
4. **Relations and functions describe the pipeline.** Observation → flux is a function, event ↔ algorithm is
   many-to-many, and event → epoch fails injectivity when a transit is split into two detections.
5. **Combinatorics drives experiments.** Cartesian products of settings, constrained counting and C(n, k)
   sampling are used. Projections decide how many expensive runs are really needed.

## Live demo script (about 5 minutes)
1. **Home → Analyse Kepler-10** (real data, bundled offline). The BLS period of 0.83754 d matches the archive
   value of 0.83749 d. The phase-folded plot shows a ~130 ppm transit: a rocky planet.
2. **Set Theory.** |A| = 39 and |B| = 102. Slide τ and watch A ∩ B change. Highlight B − A: 67 transits are too
   shallow to see one at a time.
3. **Number Theory.** Drag the period slider and watch the modular clock scatter, then snap it back. The interval
   table shows n = 2, 3 … for missed transits.
4. **Graph Theory.** Click a vertex to jump to that dip on the light curve. Show the transitivity
   counterexample.
5. **Synthetic → eclipsing-binary option** (odd/even ratio 0.5). The checklist flags a non-planetary origin.
   This is why ExoScope says *candidate*.
6. **Combinatorics.** 4 × 3 × 2 configurations, constraints remove some, run them, and show the heatmap.
7. **Candidate Results → download the HTML report**, which includes the Discrete Mathematics section.

## SDG alignment
- **Primary: SDG 9, Industry, Innovation and Infrastructure (Target 9.5: enhance scientific research and
  upgrade technological capabilities).** Open, reproducible and documented computational methods applied to
  publicly funded space-mission data. Research-grade techniques (BLS, vetting statistics) are made usable and
  inspectable by students.
- **Secondary: SDG 4, Quality Education.** Guided lessons, worked examples, quizzes with feedback and
  interactive experiments link abstract mathematics to a real scientific question.

*These are statements of alignment and intended contribution. ExoScope does not claim to achieve the UN goals.*

## Honest limitations to mention
Candidates only. Box model. One quarter/sector at a time. Checklist thresholds are conventions. The synthetic
benchmark uses white noise. Real validation covers a handful of known planets (see `docs/EVALUATION.md`).
