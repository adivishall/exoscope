# How ExoScope demonstrates Discrete Mathematics

Each concept is **implemented in code that the analysis actually uses**, shown visually in the app, and
explained with a worked example. Pure-Python implementations live in `exoscope/dmath/` (no UI code) and are
unit-tested in `tests/`.

| Module | Code | Used by the pipeline for | App page |
|---|---|---|---|
| Set theory | `dmath/sets.py` | Comparing detectors, agreement criterion, precision/recall | Set Theory |
| Number theory | `dmath/number_theory.py` | The threshold detector's period estimate, interval criterion, alias checks | Number Theory |
| Graph theory | `dmath/graphs.py` | Event network in the Lab and report | Graph Theory |
| Relations & functions | `dmath/relations.py` | Describing/validating pipeline mappings (e.g. duplicate timestamps, split events) | Relations & Functions |
| Combinatorics | `dmath/combinatorics.py`, `experiments.py` | Enumerating, constraining and executing experiments with memoisation | Combinatorics |

---

## 1. Set theory — candidate detection comparison

**Definitions.** `A` = threshold events, `B` = BLS-predicted transit epochs that contain data,
`G` = BLS-predicted epochs in data gaps, universe `U = A ∪ B ∪ G`.

**Comparing equivalent objects.** BLS outputs one periodic signal, not events. ExoScope expands it into
per-epoch events so both sets hold the same kind of element. Events are matched **one-to-one** within a
tolerance τ (default 2 h) using an optimal assignment (Hungarian algorithm, `match_events`). A matched pair
becomes one element of `U`.

**Operations implemented:** union, intersection, both differences, complement relative to `U`, symmetric
difference, Jaccard index, an inclusion–exclusion check, and De Morgan's law (unit-tested). Precision, recall
and F1 against synthetic truth `T` are computed as `|D ∩ T|`, `|D − T|` and `|T − D|`.

**Visuals:** Venn diagram, interactive "highlight a set on the light curve", set-membership timeline, and a
sweep showing how membership changes with τ.

**Example result (Kepler-10, Q3):** |A| = 39, |B| = 102, |A ∩ B| = 35, |B − A| = 67. The 67 shallow transits that
only BLS recovers are the point of the comparison.

## 2. Number theory — periodicity

- **Approximate divisibility.** `P ∣_ε Δ ⇔ ∃ n ∈ ℤ⁺: |Δ − nP| ≤ ε` (`is_approx_multiple`, `interval_table`).
  Gaps with n > 1 reveal missed transits.
- **Congruence.** Transits satisfy `t ≡ t₀ (mod P)`. `largest_congruence_class` finds the biggest set of
  events that are mutually congruent modulo a candidate P.
- **Approximate GCD (`estimate_period`).** Candidate periods are `Δᵢⱼ / k`. The score is *members × fill
  fraction*, so sub-multiples P/k, which also "divide" everything (like 1 divides every integer), lose to P.
  Then integer epochs are assigned and `tₙ = t₀ + nP` is fitted by least squares. **This is the improved threshold
  detector's period estimator.**
- **Euclid's algorithm and exact GCDs** (`euclid_steps`, `gcd_list`) on discretised integer times. These show why
  exact GCDs fail with noise.
- **Modular residues** (`modular_residues`) in chosen integer units (hours, 10 min, cadences).
- **Rational approximation** (`rational_approximation`, `alias_relation`) detects aliases (P ≈ ½·P′ or 2·P′).

**Visuals:** interval histogram with nP lines, a "modular clock" polar plot driven by a period slider, an O−C
diagram, residue tables and Euclid step tables.

## 3. Graph theory — event relationship network

**Edge rule (documented in the UI):** `{u, v} ∈ E ⇔ ∃ n ∈ {1..N}: ||t_v − t_u| − nP| ≤ ε`, optionally with a
depth-ratio limit. Implemented: vertex/edge counts, density, connected components (with a per-component period
refit), isolated vertices, maximum clique, adjacency matrix, and a **transitivity counterexample**: the
tolerance relation is reflexive and symmetric but not transitive, so it is not an equivalence relation.

**Visuals:** an interactive network (time × depth, force-directed or circular layouts). Clicking a vertex zooms
the light curve to that event and lists its neighbours with Δt, n and residual.

## 4. Relations and functions

`Relation` stores domain, codomain and ordered pairs, and checks: function, injective, surjective, bijective,
classification as one-to-one / one-to-many / many-to-one / many-to-many, a relation matrix, composition, and, for
relations on one set, reflexive, irreflexive, symmetric, antisymmetric, transitive, equivalence and orders.

Applied to real outputs:
- `f: observation → flux` is a **function**. Preprocessing removes duplicate timestamps to guarantee it.
- `R: event ↔ algorithm` is **many-to-many** and explicitly **not a function**.
- `g: event → epoch` is a function into ℤ. Non-injectivity flags split detections.
- `h: event pair → Δt`, `~` (periodic consistency, not transitive), `≺` (occurs before, a strict order),
  and `k: configuration → #detections` (many-to-one, from the combinatorics experiment).

Students can also type their own relation and have it classified.

## 5. Combinatorics — parameter combination explorer

Each setting is a set `Sᵢ`. Configurations form `S₁ × … × S_k`, counted with the rule of product.
**Constraints** (flatten window ≥ 3 × longest trial duration; longest duration < minimum period) give a
restricted count. `C(n, m)` counts the subsets that can be run, chosen as the first m or a seeded random sample.
**Projections** onto preprocessing (and preprocessing + BLS) parameters give the number of distinct expensive runs.
The runner memoises by exactly those keys, which the unit tests verify. Results are shown as a table and a
two-parameter heatmap. A "possible vs meaningful" summary counts the distinct outcomes (configuration → result
is many-to-one).
