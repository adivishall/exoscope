# ExoScope — Live Demo Guide

Every expected value below was produced by the current code on 2026-10-28. If your numbers differ in the last
digit, that is fine. If they differ a lot, use the recovery procedure at the end.

## 1. Before the presentation (do this 10 minutes early)

```bash
cd "$HOME/Projects/Discrete maths CP/exoscope"
.venv/bin/streamlit run app.py --server.port 8501 --server.address localhost
```

- Open http://localhost:8501. The **first page load takes ~10 s** while Python imports Astropy, so do it before
  the audience is watching.
- Click **Analyse Kepler-8** once to warm everything up, then click **Home**.
- No environment variables, API keys or accounts are needed. The bundled samples work **without internet**.
- If port 8501 is busy (another Streamlit is running), stop it with **Ctrl+C** in its terminal or use
  `--server.port 8502`.
- Projector tip: the browser zoom (Cmd + / Cmd −) is safe. Metric rows wrap automatically on narrow screens.

## 2. Recommended sequence (~7 minutes)

| # | Where | Do | Expected (verified) | Say |
|---|---|---|---|---|
| 1 | Home | Read the info box, click **Analyse Kepler-10** | Goes to Analyze. BLS period **0.83754 d**, depth **134 ppm**, SNR **43.5**; threshold events **39** | "A rocky planet — each transit is only ~130 ppm, about 2× the per-point noise of 59 ppm." |
| 2 | Analyze → **BLS** tab | Scroll to the periodogram and folded curve | Peak at 0.8375 d on the dashed archive line (Kepler-10 b, 0.83749 d); a clear box-shaped dip in the folded plot | "Folding is time mod P — 102 transits stacked into one." |
| 3 | Set Theory | Scroll to ③ Your data | \|A\| = 39, \|B\| = 102, \|A∩B\| = 35, \|A−B\| = 4, \|B−A\| = 67, gaps (A∪B)ᶜ = 5, Jaccard 0.33; inclusion–exclusion holds | "B − A are the 67 transits too shallow for a per-point threshold; BLS adds them up." Choose **B − A** in the radio to highlight them. |
| 4 | Set Theory | Drag **τ** from 2 h to 0.25 h | \|A∩B\| drops | "Set membership depends on the matching rule we chose." |
| 5 | Number Theory | Drag the **Trial period** slider away and back | Modular clock scatters, then re-clusters; 32/38 intervals ≈ n·P at the BLS period | "Gaps with n = 2, 3 … are missed transits. The period is an approximate GCD." |
| 6 | Number Theory → Approximate GCD & O−C tab | — | Approximate-GCD period **0.83748 d** (from threshold events alone) vs BLS 0.83754 d — ratio 1 (equal) | "The threshold detector's period comes from this number-theory method." |
| 7 | Graph Theory | Look at the metrics and click any vertex | \|V\| = 39, \|E\| = 111, 6 components, 4 isolated, largest clique 7; the linked view zooms to that dip | "Isolated vertices are dips no other event supports — likely noise." Show the **transitivity counterexample** box. |
| 8 | Relations & Functions | Pick **R: Event ↔ Algorithm** | Type **many-to-many**, Function ❌, with the reason listed | "An event can be found by both algorithms, so this is a relation, not a function." |
| 9 | Combinatorics | Leave defaults, press **Run 12 configuration(s)** | 4 × 3 × 2 = 24 configurations, 12 valid; runs finish in a few seconds with 1 preprocessing run | "Constraints removed the 0.5-day window; projection means only one detrending run was needed." |
| 10 | Explore Targets → Synthetic generator | Open **More options**, set odd/even ratio **0.5**, click Generate, then **Analyze → Run analysis** | BLS period **≈ 6.0 d** (twice the true 3 d); verdict "signs of a non-planetary origin"; secondary eclipse ❌ | "Number theory says 3 d, BLS says 6 d — a 2:1 alias plus a deep secondary eclipse: an eclipsing binary, not a planet. That's why we say *candidate*." |
| 11 | Candidate Results | Download **Report (HTML)** | A report with charts and a Discrete Mathematics section | "Everything is exportable and saved in History." |

**Short version (3 minutes):** steps 1, 3, 5, 7, 10.

### Other verified datasets
| Dataset | Result |
|---|---|
| Kepler-8 (bundled) | P = 3.52307 d (archive 3.52250, 0.016% error), depth 7,046 ppm, \|A\| = \|B\| = \|A∩B\| = 26, Jaccard 1.00, 10/10 checks |
| WASP-18 (bundled TESS) | P = 0.94160 d (archive 0.94145), 9/10 checks — secondary-eclipse check ⚠️ (a small but significant dip at phase 0.5, 7% of the primary; hot Jupiters like WASP-18 b can show a real secondary eclipse and phase variation, which is why this is “caution”, not “fail”) |
| Synthetic default | True P = 3 d: BLS 2.99743 d, threshold/approximate-GCD 3.00009 d; F1 threshold 0.94, BLS 1.00 |
| Kepler-5 Q2 (live MAST download) | P = 3.54862 d (archive 3.54847), 10/10 checks |

## 3. Optional: live MAST download (needs internet)

Explore Targets → type `Kepler-5`, mission Kepler → **Search MAST** (~30–40 s the first time) → choose
`Quarter 2` → **Download and load** (~20–30 s first time, instant afterwards). Already cached on this laptop (search results + the one
quarter/sector used so far): Kepler-5 Q2, Kepler-6, Kepler-8, Kepler-10, Kepler-17, HAT-P-7, TrES-2, WASP-121, KELT-11. **Only do this if the
Wi-Fi is good** — otherwise say "this uses the bundled offline copy of the same archive data".

## 4. Known risks and safe workarounds

| Risk | Workaround |
|---|---|
| No internet / MAST slow | Use Home quick-start buttons or *Bundled real samples* — they are real Kepler/TESS data stored offline |
| First load slow (~10 s) | Warm up before presenting (section 1) |
| A page shows "No light curve loaded yet" | Opening a page by URL starts a new session. Click **Quick start: analyse Kepler-8** on that page |
| A Lab page looks empty after loading new data | Press **Run analysis** on Analyze, or the "Run analysis with default settings" button shown on the page |
| Graph vertex click doesn't select | Use the **Vertex** dropdown under the graph — same linked view |
| HTML report charts missing offline | Charts load Plotly from a CDN; the Markdown report and CSV/JSON exports work offline |

## 5. Recovery procedure if the app fails mid-demo

1. In the terminal press **Ctrl+C**, then rerun:
   `.venv/bin/streamlit run app.py --server.port 8501 --server.address localhost`
2. Reload the browser tab and click **Analyse Kepler-10** (or Kepler-8) on Home — you are back in ~15 s.
3. If Python packages are broken: `rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
   (needs internet, ~5–10 min).
4. Last resort: open `docs/EVALUATION.md` and the HTML report you exported in advance, and present the results
   from those. **Export one HTML report before the presentation** (step 11) as a backup.
