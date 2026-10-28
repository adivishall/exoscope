"""Static teaching content: glossary, formulas and guided lessons."""

from __future__ import annotations

GLOSSARY: list[dict[str, str]] = [
    # astronomy
    {"term": "Transit", "area": "Astronomy",
     "definition": "A planet passing in front of its star as seen from Earth, blocking a little starlight.",
     "example": "Kepler-8 b blocks ~0.9% of its star's light for about 3.2 hours every 3.52 days."},
    {"term": "Light curve", "area": "Astronomy",
     "definition": "Brightness (flux) of a star measured repeatedly over time.",
     "example": "A Kepler quarter contains ~4,400 measurements taken every 29.4 minutes."},
    {"term": "Flux", "area": "Astronomy", "definition": "The amount of light received per unit time.",
     "example": "ExoScope uses PDCSAP flux: the mission pipeline's systematics-corrected brightness."},
    {"term": "Normalisation", "area": "Data processing",
     "definition": "Dividing the flux by its median so the typical level is 1.0 and dips read as fractions.",
     "example": "A value of 0.991 means 0.9% dimmer than usual."},
    {"term": "Detrending (flattening)", "area": "Data processing",
     "definition": "Removing slow brightness changes (spots, rotation, drifts) while keeping short dips.",
     "example": "A 1-day Savitzky–Golay window removes a 6-day stellar wobble but keeps a 3-hour transit."},
    {"term": "Outlier / sigma clipping", "area": "Data processing",
     "definition": "Removing points further than k standard deviations from the median.",
     "example": "Only upward outliers are clipped by default — downward clipping can delete transits."},
    {"term": "Transit depth (δ)", "area": "Astronomy",
     "definition": "The fractional drop in brightness during transit, often in ppm (parts per million).",
     "example": "1% = 10,000 ppm; an Earth-size planet around a Sun-like star gives ~84 ppm."},
    {"term": "Orbital period (P)", "area": "Astronomy", "definition": "Time between consecutive transits.",
     "example": "WASP-18 b: P = 0.941 days."},
    {"term": "Epoch", "area": "Astronomy",
     "definition": "The integer cycle number n of a transit: tₙ = t₀ + n·P.",
     "example": "If t₀ = 100.0 and P = 3, the transit at 109.0 is epoch n = 3."},
    {"term": "Box Least Squares (BLS)", "area": "Algorithms",
     "definition": "Fits a periodic box-shaped dip at many trial periods and reports how well each fits.",
     "example": "BLS finds Kepler-10 b's 134 ppm transits even though each one is below the noise."},
    {"term": "Periodogram", "area": "Algorithms", "definition": "BLS power as a function of trial period.",
     "example": "Peaks also appear at P/2 and 2P (aliases)."},
    {"term": "SNR", "area": "Statistics",
     "definition": "Signal-to-noise ratio: signal size divided by its uncertainty.",
     "example": "depth / σ · √(points in transit)."},
    {"term": "SDE", "area": "Statistics",
     "definition": "Signal detection efficiency: how far the highest periodogram peak stands above the rest, in standard deviations.",
     "example": "SDE ≈ 27 for the Kepler-8 sample."},
    {"term": "Phase folding", "area": "Algorithms",
     "definition": "Plotting every point at (t − t₀) mod P so all transits stack on top of each other.",
     "example": "The folded light curve is the modular-arithmetic view of the data."},
    {"term": "Eclipsing binary", "area": "Astronomy",
     "definition": "Two stars orbiting and eclipsing each other — the most common transit false positive.",
     "example": "Shows unequal odd/even depths or a deep secondary eclipse."},
    {"term": "Candidate", "area": "Astronomy",
     "definition": "A signal consistent with a planet that has not been independently confirmed.",
     "example": "Every ExoScope detection is a candidate."},
    # discrete mathematics
    {"term": "Set", "area": "Discrete math", "definition": "An unordered collection of distinct elements.",
     "example": "A = {T1, T2, T3} — the threshold detections."},
    {"term": "Union / intersection / difference", "area": "Discrete math",
     "definition": "A ∪ B: in A or B. A ∩ B: in both. A − B: in A but not B.",
     "example": "A ∩ B = events found by both detectors."},
    {"term": "Complement", "area": "Discrete math",
     "definition": "Aᶜ = U − A: everything in the universe U that is not in A.",
     "example": "With U = all predicted epochs, (A ∪ B)ᶜ = epochs that fell in data gaps."},
    {"term": "Inclusion–exclusion", "area": "Discrete math", "definition": "|A ∪ B| = |A| + |B| − |A ∩ B|.",
     "example": "39 + 102 − 35 = 106 for the Kepler-10 sample."},
    {"term": "Divisibility", "area": "Discrete math",
     "definition": "a ∣ b if b = n·a for an integer n. With tolerance: |b − n·a| ≤ ε.",
     "example": "An interval of 7.04 d is ≈ 2 × 3.52 d: one transit was missed."},
    {"term": "Congruence (mod)", "area": "Discrete math",
     "definition": "a ≡ b (mod m) when a − b is a multiple of m (same remainder).",
     "example": "All transit times are congruent to t₀ modulo P."},
    {"term": "GCD", "area": "Discrete math",
     "definition": "Greatest common divisor — the largest number dividing all given numbers.",
     "example": "The period is the (approximate) GCD of the intervals between transits."},
    {"term": "Graph", "area": "Discrete math",
     "definition": "G = (V, E): vertices joined by edges.",
     "example": "Vertices = events; an edge joins two events whose separation is a whole number of periods."},
    {"term": "Connected component", "area": "Discrete math",
     "definition": "A maximal set of vertices reachable from each other along edges.",
     "example": "The largest component collects the events consistent with one period."},
    {"term": "Clique", "area": "Discrete math", "definition": "A set of vertices that are all pairwise adjacent.",
     "example": "A clique of events is pairwise consistent with one period."},
    {"term": "Relation", "area": "Discrete math", "definition": "Any subset R ⊆ X × Y of ordered pairs.",
     "example": "(T3, Threshold), (T3, BLS) ∈ R: event T3 was found by both."},
    {"term": "Function", "area": "Discrete math",
     "definition": "A relation where every x ∈ X has exactly one y.",
     "example": "Observation time → flux."},
    {"term": "Injective / surjective / bijective", "area": "Discrete math",
     "definition": "Injective: no two inputs share an output. Surjective: every output is hit. Bijective: both.",
     "example": "Event → epoch is injective unless one transit was split into two events."},
    {"term": "Equivalence relation", "area": "Discrete math",
     "definition": "Reflexive, symmetric and transitive; it partitions a set into classes.",
     "example": "'Periodically consistent within ε' is NOT transitive, so it is not one."},
    {"term": "Cartesian product", "area": "Discrete math",
     "definition": "S₁ × S₂ = all ordered pairs (a, b) with a ∈ S₁, b ∈ S₂.",
     "example": "{2.5, 3} × {1, 2} = {(2.5,1), (2.5,2), (3,1), (3,2)}."},
    {"term": "Rule of product", "area": "Discrete math",
     "definition": "|S₁ × … × S_k| = |S₁| · … · |S_k|.",
     "example": "4 thresholds × 3 min-points × 2 windows = 24 configurations."},
    {"term": "Combination C(n, k)", "area": "Discrete math",
     "definition": "Number of ways to choose k items from n, order ignored: n! / (k!(n−k)!).",
     "example": "Choosing 10 of 24 configurations: C(24, 10) = 1,961,256 ways."},
]

FORMULAS: list[dict[str, str]] = [
    {"name": "Transit depth", "latex": r"\delta \approx \left(\frac{R_p}{R_s}\right)^2",
     "note": "Assumes a dark planet fully in front of a uniformly bright star (no limb darkening, no grazing "
             "geometry, no light from other stars). It gives a radius *ratio*; a planet radius also needs the "
             "star's radius, and is only approximate."},
    {"name": "Planet radius from depth", "latex": r"R_p \approx R_s\sqrt{\delta}",
     "note": "With R☉ ≈ 109.1 R⊕. Example: δ = 134 ppm, Rs = 1.06 R☉ → Rp ≈ 1.3 R⊕."},
    {"name": "Linear ephemeris", "latex": r"t_n = t_0 + n\,P,\qquad n \in \mathbb{Z}",
     "note": "Every transit time is the reference time plus an integer number of periods."},
    {"name": "Approximate divisibility", "latex": r"P \mid_\varepsilon \Delta \iff \exists\, n\in\mathbb{Z}^+ : |\Delta - nP| \le \varepsilon",
     "note": "Used to test whether an interval is a whole number of orbits."},
    {"name": "Phase (modular arithmetic)", "latex": r"\phi = \frac{(t - t_0) \bmod P}{P}",
     "note": "Folding the light curve is reducing time modulo P."},
    {"name": "Expected duration (Sun-like star)", "latex": r"T_{dur} \approx 13\,\text{h}\left(\frac{P}{365\,\text{d}}\right)^{1/3}\left(\frac{\rho_\star}{\rho_\odot}\right)^{-1/3}",
     "note": "Central transit, circular orbit; a rough plausibility check only."},
    {"name": "Signal-to-noise", "latex": r"\text{SNR} = \frac{\delta}{\sigma}\sqrt{N_{in}}",
     "note": "σ is the per-point noise, N_in the number of in-transit points."},
    {"name": "Robust noise", "latex": r"\sigma \approx 1.4826\cdot\text{median}(|x - \text{median}(x)|)",
     "note": "The median absolute deviation, scaled to match a standard deviation for Gaussian noise."},
    {"name": "Inclusion–exclusion", "latex": r"|A\cup B| = |A| + |B| - |A\cap B|", "note": ""},
    {"name": "Jaccard similarity", "latex": r"J(A,B) = \frac{|A\cap B|}{|A\cup B|}", "note": "1 = identical sets, 0 = disjoint."},
    {"name": "Precision, recall, F1", "latex": r"P = \frac{|D\cap T|}{|D|},\; R = \frac{|D\cap T|}{|T|},\; F_1 = \frac{2PR}{P+R}",
     "note": "D = detections, T = true transits (only known for synthetic data)."},
    {"name": "Rule of product", "latex": r"|S_1\times\cdots\times S_k| = \prod_{i=1}^{k}|S_i|", "note": ""},
    {"name": "Combinations", "latex": r"\binom{n}{k} = \frac{n!}{k!\,(n-k)!}", "note": ""},
]

LESSONS: list[dict] = [
    {
        "id": "transits",
        "title": "1 · What is a transit?",
        "steps": [
            "**A star's brightness is a time series.** A space telescope measures the star every few minutes. "
            "Plotting brightness against time gives a *light curve*.",
            "**A planet crossing the star blocks light.** The dip depth tells us the size ratio: "
            "δ ≈ (Rp/Rs)². A Jupiter around the Sun gives ~1%, an Earth only ~0.008%.",
            "**Repetition is the key evidence.** One dip could be anything. Dips that repeat with the same period, "
            "depth and duration point to an orbiting body — but eclipsing binary stars can mimic this.",
            "**Try it:** open *Explore Targets → Bundled samples*, load Kepler-8 and look at the raw light curve. "
            "Can you spot the dips before any processing?",
        ],
        "learned": "Transits are periodic, shallow dips. Detecting them is about finding repetition in noisy data — "
                   "which is where discrete mathematics helps.",
    },
    {
        "id": "sets",
        "title": "2 · Sets: did the two detectors agree?",
        "steps": [
            "Let **A** be the events the threshold detector found and **B** the transit epochs predicted by BLS.",
            "Worked example: A = {E1, E2, E3, E5}, B = {E2, E3, E4, E5, E6}. Then A ∩ B = {E2, E3, E5} (agreement), "
            "A − B = {E1} (threshold only), B − A = {E4, E6} (BLS only), A ∪ B has 4 + 5 − 3 = 6 elements.",
            "**Matching needs a tolerance.** Two detectors never report identical times, so we treat events within "
            "±τ hours as the *same element*. Changing τ changes the sets!",
            "**Try it:** in *Set Theory*, slide the tolerance and watch |A ∩ B| and the Jaccard index change.",
        ],
        "learned": "Set operations turn 'do the methods agree?' into exact counts. B − A often contains shallow "
                   "transits that only BLS can see.",
    },
    {
        "id": "numbers",
        "title": "3 · Number theory: finding the period",
        "steps": [
            "Transit times follow tₙ = t₀ + n·P. So the gap between any two transits is an **integer multiple** of P.",
            "Worked example (hours): events at 10, 34, 58, 106, 130. Gaps: 24, 24, 48, 24. "
            "gcd(24, 24, 48, 24) = 24 → P = 24 h. The 48 h gap means one transit (at 82 h) was missed.",
            "**Noise breaks exact GCDs.** With gaps 24.4, 23.4, 48.5, 23.6 the integer GCD is meaningless. We use "
            "*approximate divisibility*: |Δ − n·P| ≤ ε.",
            "**Congruence view:** all transit times have the same remainder modulo P. On a 'modular clock' they "
            "line up at the same angle. A wrong P scatters them around the circle.",
            "**Try it:** in *Number Theory*, drag the period slider and watch the clock.",
        ],
        "learned": "The period is the approximate greatest common divisor of the intervals; missed transits show "
                   "up as multiples n > 1, and false events as points that are not congruent to t₀.",
    },
    {
        "id": "graphs",
        "title": "4 · Graphs: a network of consistent events",
        "steps": [
            "Make each event a **vertex**. Join two vertices by an **edge** if their separation is a whole number of "
            "periods (within ε).",
            "Events from one planet form a **connected component** — usually a near-clique. Random noise events are "
            "often **isolated vertices**.",
            "Surprise: 'consistent with' is **not transitive**. If u~v and v~w, the small errors can add up so that "
            "u and w are not consistent. That is why components (chains) and cliques (all pairs) differ.",
            "**Try it:** in *Graph Theory*, tighten the tolerance until the big component splits.",
        ],
        "learned": "Graph structure summarises which events support each other; connectivity is evidence for "
                   "periodicity, not proof of a planet.",
    },
    {
        "id": "relations",
        "title": "5 · Relations and functions in the pipeline",
        "steps": [
            "A **relation** is a set of ordered pairs. A **function** gives each input exactly one output.",
            "Observation time → flux is a function. Event ↔ algorithm is **many-to-many**: an event can be found by "
            "both detectors, and a detector finds many events — so it is *not* a function.",
            "Event → epoch number is a function into ℤ; if two events get the same epoch it is not injective, which "
            "reveals a transit split into two detections.",
            "**Try it:** in *Relations & Functions*, build your own relation and let ExoScope classify it.",
        ],
        "learned": "Relations describe the structure of the pipeline; checking 'is this a function?' catches real "
                   "data problems such as duplicate timestamps or split events.",
    },
    {
        "id": "combinatorics",
        "title": "6 · Combinatorics: systematic experiments",
        "steps": [
            "Every setting is a set of choices. A full configuration is an element of the Cartesian product.",
            "Rule of product: 4 thresholds × 3 minimum-point values × 2 windows = 24 configurations.",
            "Constraints remove invalid ones (e.g. a detrending window shorter than 3 transit durations would erase "
            "the transit). Choosing 10 of the 24 to run can be done in C(24, 10) ways.",
            "Many configurations give the *same* answer — the configuration → result function is many-to-one. "
            "Possible configurations ≠ meaningful results.",
            "**Try it:** in *Combinatorics*, pick values, read the count, then run them and compare.",
        ],
        "learned": "Counting tells you how big an experiment is before you run it, and projections show how much "
                   "work can be shared.",
    },
]
