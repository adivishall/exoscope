"""Multiple-choice quiz: a fixed bank plus questions built from the current analysis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class Question:
    qid: str
    topic: str
    prompt: str
    options: list[str]
    answer: int  # index into options
    explanation: str
    dynamic: bool = False
    tags: list[str] = field(default_factory=list)

    def check(self, choice: int | None) -> bool:
        return choice == self.answer


STATIC_QUESTIONS: list[Question] = [
    Question("s1", "Sets", "A = {E1, E2, E3}, B = {E2, E3, E4, E5}. What is A − B?",
             ["{E1}", "{E4, E5}", "{E2, E3}", "{E1, E4, E5}"], 0,
             "A − B keeps the elements of A that are not in B: only E1."),
    Question("s2", "Sets", "Why does ExoScope need a matching tolerance before comparing detections as sets?",
             ["Two methods never report exactly the same timestamp for one transit",
              "Sets cannot contain decimal numbers", "To make the union smaller", "BLS needs it to run"], 0,
             "Equal elements must be identified first; times within ±τ are treated as the same event."),
    Question("s3", "Sets", "|A| = 10, |B| = 14, |A ∩ B| = 8. What is |A ∪ B|?", ["16", "24", "22", "8"], 0,
             "Inclusion–exclusion: 10 + 14 − 8 = 16."),
    Question("n1", "Number theory", "Transits at 2.0, 5.0, 11.0, 14.0 days. Which period best explains all gaps?",
             ["3 days", "6 days", "9 days", "1.5 days"], 0,
             "Gaps are 3, 6, 3: all multiples of 3. The 6-day gap means one transit (8.0) was missed. 1.5 also divides "
             "them but predicts many transits that never appear."),
    Question("n2", "Number theory", "With P = 4 d, an interval of 12.1 d with tolerance 0.2 d corresponds to…",
             ["n = 3 (two transits missed between)", "n = 12", "no integer multiple", "n = 4"], 0,
             "12.1 / 4 ≈ 3.03 → n = 3, residual 0.1 d ≤ 0.2 d. n − 1 = 2 transits were not detected in between."),
    Question("n3", "Number theory", "What does 'all transit times are congruent modulo P' mean?",
             ["They leave the same remainder when reduced mod P",
              "They are all equal", "They are all divisible by P", "They are all prime"], 0,
             "t ≡ t₀ (mod P): (t − t₀) is an integer multiple of P, so every transit has the same phase."),
    Question("n4", "Number theory", "Why can't we just use the exact integer GCD of measured intervals?",
             ["Noise makes intervals slightly off, so the exact GCD collapses to a tiny meaningless value",
              "GCD only works for prime numbers", "Intervals are always integers", "It is too slow"], 0,
             "e.g. gcd(24, 23, 48) = 1. Approximate divisibility with a tolerance is required."),
    Question("g1", "Graph theory", "In the event graph, an isolated vertex most likely represents…",
             ["An event not consistent with any other event at this period (often noise)",
              "The strongest transit", "The first observation", "A data gap"], 0,
             "No edge means no other event lies a whole number of periods away within the tolerance."),
    Question("g2", "Graph theory", "Why is 'periodically consistent within ε' not an equivalence relation?",
             ["It is not always transitive: small errors can accumulate along a chain",
              "It is not reflexive", "It is not symmetric", "Graphs cannot represent relations"], 0,
             "u~v and v~w can each be within ε while u and w differ by more than ε from a whole number of periods."),
    Question("g3", "Graph theory", "A graph has 5 vertices and every pair is connected. How many edges?",
             ["10", "5", "20", "25"], 0, "A complete graph K₅ has C(5, 2) = 10 edges."),
    Question("r1", "Relations", "R = {(E1, Threshold), (E1, BLS), (E2, BLS)}. Is R a function from events to algorithms?",
             ["No — E1 is related to two algorithms", "Yes — every event has an algorithm",
              "Yes — it is many-to-one", "No — E2 has no image"], 0,
             "A function needs exactly one output per input; E1 has two."),
    Question("r2", "Relations", "Planet → host star (Kepler-10 b ↦ Kepler-10, Kepler-10 c ↦ Kepler-10) is…",
             ["A function that is not injective", "Not a function", "A bijection", "One-to-many"], 0,
             "Each planet has one host (function); two planets share a host (not injective) — many-to-one."),
    Question("r3", "Relations", "The relation 'event u occurs before event v' is…",
             ["A strict order: irreflexive, antisymmetric, transitive", "An equivalence relation",
              "Reflexive and symmetric", "Not a relation"], 0,
             "No event precedes itself, never both u≺v and v≺u, and u≺v≺w implies u≺w."),
    Question("c1", "Combinatorics", "3 thresholds × 2 windows × 4 minimum-point values give how many configurations?",
             ["24", "9", "12", "64"], 0, "Rule of product: 3 · 2 · 4 = 24."),
    Question("c2", "Combinatorics", "Out of 20 valid configurations you run 5. How many different 5-sets could you pick?",
             ["15,504", "100", "3,125", "1,860,480"], 0, "C(20, 5) = 15,504 (order does not matter)."),
    Question("c3", "Combinatorics", "Why does a larger configuration count not mean more scientific results?",
             ["Many configurations give the same answer and some are physically invalid",
              "Computers cannot count that high", "Each configuration finds a different planet",
              "Combinatorics only applies to integers"], 0,
             "The configuration → result map is many-to-one, and constraints rule some configurations out."),
    Question("a1", "Astronomy", "A transit depth of 1% implies Rp/Rs of about…", ["0.1", "0.01", "0.5", "1"], 0,
             "δ ≈ (Rp/Rs)² → Rp/Rs ≈ √0.01 = 0.1 (Jupiter-size around a Sun-like star)."),
    Question("a2", "Astronomy", "Odd and even transits have clearly different depths. This suggests…",
             ["An eclipsing binary at twice the period", "A smaller planet", "Better data quality", "A longer period planet"], 0,
             "Two stars produce alternating primary and secondary eclipses of different depth."),
]


def _shuffle_with_answer(correct: str, wrong: list[str], seed: int) -> tuple[list[str], int]:
    opts = [correct] + [w for w in dict.fromkeys(wrong) if w != correct][:3]
    k = seed % len(opts)
    opts = opts[k:] + opts[:k]
    return opts, opts.index(correct)


def dynamic_questions(result) -> list[Question]:
    """Questions whose numbers come from the analysis currently loaded."""
    qs: list[Question] = []
    if result is None:
        return qs
    comp = result.comparison
    s = comp.summary()
    if comp.bls_available:
        correct = s["|A∪B|"]
        opts, ans = _shuffle_with_answer(str(correct), [str(s["|A|"] + s["|B|"]), str(s["|A∩B|"]),
                                                        str(abs(s["|A|"] - s["|B|"]) + 1)], correct)
        qs.append(Question("d_union", "Sets",
                           f"In your current analysis |A| = {s['|A|']}, |B| = {s['|B|']} and |A ∩ B| = {s['|A∩B|']}. "
                           "What is |A ∪ B|?", opts, ans,
                           f"|A| + |B| − |A ∩ B| = {s['|A|']} + {s['|B|']} − {s['|A∩B|']} = {correct}.", dynamic=True))
    best = result.best
    thr = result.threshold
    if best and thr and len(thr.events) >= 2:
        from ..dmath.number_theory import interval_table

        rows = [r for r in interval_table([e.t_center for e in thr.events], best.period,
                                          result.settings.match_tolerance_hours / 24) if r["consistent"]]
        multi = [r for r in rows if r["n"] > 1] or rows
        if multi:
            r = multi[0]
            correct = r["n"] - 1
            opts, ans = _shuffle_with_answer(str(correct), [str(correct + 1), str(r["n"]), str(max(correct - 1, 0) + 3)],
                                             r["n"])
            qs.append(Question("d_missed", "Number theory",
                               f"Between threshold events {r['pair']} the interval is {r['delta_days']:.4f} d and "
                               f"BLS gives P = {best.period:.4f} d. How many transits were missed in between?",
                               opts, ans,
                               f"Δ/P = {r['ratio']:.3f} ≈ n = {r['n']}, so {correct} transit(s) lie between them "
                               "undetected (or in a gap).", dynamic=True))
    if best:
        frac = Fraction(best.period / (best.period * 2)).limit_denominator(4)
        qs.append(Question("d_alias", "Number theory",
                           f"The BLS periodogram may also peak at {2 * best.period:.4f} d. How is it related to "
                           f"P = {best.period:.4f} d?",
                           ["It is 2P, an integer multiple (alias)", "It is P/2", "It is unrelated", "It is P²"], 0,
                           f"{2 * best.period:.4f} / {best.period:.4f} = 2 (P / 2P = {frac}): folding at 2P stacks "
                           "every second transit, so BLS also scores it well.", dynamic=True))
    n_events = len(thr.events) if thr else 0
    if n_events >= 2:
        pairs = math.comb(n_events, 2)
        opts, ans = _shuffle_with_answer(f"{pairs:,}", [f"{n_events ** 2:,}", f"{n_events - 1:,}", f"{2 * n_events:,}"],
                                         n_events)
        qs.append(Question("d_pairs", "Graph theory",
                           f"Your threshold detector found {n_events} events. How many possible edges (unordered "
                           "pairs) could the event graph have at most?", opts, ans,
                           f"C({n_events}, 2) = {n_events}·{n_events - 1}/2 = {pairs:,}.", dynamic=True))
    return qs


def shuffle_options(q: Question, seed: int) -> Question:
    """Return a copy with options in a reproducible random order (seeded by question id + attempt)."""
    perm = list(range(len(q.options)))
    random.Random(f"{q.qid}-{seed}").shuffle(perm)
    return Question(q.qid, q.topic, q.prompt, [q.options[i] for i in perm], perm.index(q.answer),
                    q.explanation, q.dynamic, list(q.tags))


def build_quiz(result=None, topics: list[str] | None = None, seed: int = 0) -> list[Question]:
    qs = dynamic_questions(result) + STATIC_QUESTIONS
    if topics:
        qs = [q for q in qs if q.topic in topics]
    return [shuffle_options(q, seed) for q in qs]


def score(questions: list[Question], answers: dict[str, int | None]) -> tuple[int, int]:
    return sum(q.check(answers.get(q.qid)) for q in questions), len(questions)
