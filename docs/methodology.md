# Methodology

## Problem
Given incidents (people, priority 1–5, road, access constraints), teams, vehicles (type, capacity per trip, max trips) and shelters (capacity), produce a plan: team assignments, vehicle assignments (vehicle, incident, trips, people), shelter allocations and the unserved demand.

## Hard constraints (validator rules)
| Rule | Type | Check |
|---|---|---|
| V1 | vehicle_capacity | people ≤ trips × capacity per assignment; Σ trips of a vehicle ≤ `max_trips` |
| V2 | team_availability | only available teams; distinct incidents per team ≤ `max_incidents` |
| V3 | shelter_capacity | Σ people allocated to a shelter ≤ capacity |
| V4 | road_accessibility | non-boat vehicles may not serve an incident on a blocked road |
| V5 | coverage | transported ≤ people; transported = sheltered; unserved reported exactly; transported sites need a team |
| V6 | compatibility | vehicle type ∈ incident's allowed types |
| V7 | schema_integrity | known ids, no duplicates, no negatives, parseable output |

Records with unknown ids are reported by V7 only, to avoid cascading duplicate violations.

## Scenario generation
`generate_scenario(size, scarcity, seed)` uses `random.Random` seeded from the arguments, so output is reproducible. Sizes (incidents/teams/vehicles/shelters): small 4/3/6/3, medium 8/6/12/5, large 14/10/20/8. Scarcity scales transport and shelter capacity relative to demand (low 1.6×, medium 1.0×, high ≈0.6×) and the probability a team is unavailable (5/15/25 %). Each scenario stores a content hash and difficulty metadata (`difficulty_score` is a heuristic for grouping, not a ground truth). The data are **synthetic**; they illustrate constraint structure, not any real region.

## Difficulty levels
`generate_level_scenario(level, seed)` reuses the same generator and adds optional knobs. A level is a named bundle: size profile, scarcity profile, blocked-road probability, plus `compat_restrict_prob` (each allowed vehicle type is removed from an incident with this probability, keeping one usable type), `boat_cap` (fleet has at most this many boats, which are the only way into blocked-road sites), `team_slot_ratio` (usable team slots ≈ ratio × incidents), `shelter_skew` (share of shelter capacity in one shelter) and `interlock_prob` (a blocked-road site is reachable only by boat).

| Level | Size | Scarcity | Blocked roads | Extras |
|---|---|---|---|---|
| L0 | small | low | 0.2 | none (easy baseline) |
| L1 | medium | medium | 0.2 | none |
| L2 | large | medium | 0.4 | none |
| L3 | large | medium | 0.3 | restrictions 0.35, ≤2 boats, team slots 0.7×incidents |
| L4 | large | high | 0.4 | restrictions 0.35, ≤2 boats, team slots 0.6×incidents, one shelter with 40 % of capacity, boat-only sites 0.7 |

Scenarios made without a level are unchanged (same seeds give the same content hash as before). Level scenarios have id `L<k>-s<seed>`. A solvability guard repairs a level scenario until at least 70 % of incidents can be served by some vehicle type that exists in the fleet; `difficulty.servable_incidents` and `servable_demand_fraction` record the result. (Note: an empty plan is always feasible, so "solvable" here means "most demand is reachable", not "has a feasible plan".)

**These levels are design choices, not measured difficulty.** In a sandbox probe with a stand-in MIP solver (12 seeds per level, no LLM), a simple greedy heuristic reached roughly 94–99 % of the solver's objective on average at every level, so the levels mostly raise the *number and interaction of rules* rather than the optimisation difficulty. Whether they are hard for an LLM is an empirical question for the pilot. Re-run `experiments/scripts/probe_levels.py` with real OR-Tools to see your own numbers.

## Methods
- **M1 LLM-only** — one call; validator used only for scoring.
- **M2 Self-critique** — critique call + revision call per round; the model decides when it is done; validator used only for scoring.
- **M3 ResQ-Verify** — validate → structured feedback → revision, until valid or the budget is spent.
- **M5 Resample + Verify** — validate; if infeasible, make a fresh independent call with the *unchanged* planner prompt (new per-call seed, same model, temperature and schema). The model never sees violations; the validator only decides whether to stop. The final plan is the first feasible one, or the last attempt.
- **M2 and M3 share one revision template** (`prompts/revision.txt`, placeholder `{{FEEDBACK}}`): M2 fills it with the model's own critique, M3 with the validator report. Instructions, output schema, model and temperature are identical, so the comparison isolates the *source* of feedback. Per-template hashes are stored with every run (`prompt_template_hashes`, `revision_template_hash`).

## Call budgets
Budget B = maximum LLM calls per run. M1: 1. M3, M5: first plan + up to B−1 more. M2: first plan + ⌊(B−1)/2⌋ rounds of 2 calls (a lone leftover call is not spent). Runs stop early at the first valid plan (M3, M5) or when the critique approves (M2). The first plan is shared across methods (`shared_initial_plan`) but counted as one call in every method's `calls_used`. With `nested_budgets` a run at a smaller budget is read off the larger run's first rounds (identical to running it separately when the model's replies are a function of the prompt and seed, which a test checks with the simulated LLM; real Gemini replies are not guaranteed to repeat, so nested results are one sample path per scenario).
- **M4 OR-Tools CP-SAT** — exact model (variables trips, people, team assignment, placement; maximise `1000·Σ priority·served − Σtrips − Σassignments`). Time-limited, so the status (OPTIMAL/FEASIBLE) and gap are recorded. It is a *reference* and reports its own runtime; it is not an LLM.

## Metrics
- Feasibility rate (final plan has zero violations), initial feasibility, violation rate and per-type taxonomy.
- Repair success = among runs whose initial plan was infeasible, share that ended feasible; rounds used; edit distance between rounds.
- Plan quality = Σ priority·served / Σ priority·people, with served = min(transported, sheltered, people). `effective_quality` counts infeasible plans as 0.
- Optimality gap = (CP-SAT objective − plan objective) / CP-SAT objective, for feasible plans. Negative values can occur because the solver is time-limited.
- Latency (including the initial call), tokens, and cost (only if you supply prices). Token counts are stored as empty when the API did not report them; they are never filled with 0 or estimated by this code (the mock LLM labels its own counts as estimated).
- Calls used, first feasible call (cumulative LLM calls at the first valid plan) and feasibility versus call budget.
- Failure persistence: among runs whose first plan was infeasible, the share of the initial violations (same type and entity) still present in the final plan, and per round the share of the previous round's violations still present.
- Scalability: all metrics split by scenario size.

## Statistics
- Rates: Wilson 95 % CI. Continuous metrics: mean, median, SD, seeded bootstrap 95 % CI.
- Paired comparisons use the **scenario** as the unit (repetitions averaged), so repeats are never counted as independent samples.
  Binary outcomes → exact McNemar. Continuous → Shapiro–Wilk on differences; paired t only if n ≥ 20 and not rejected, else Wilcoxon signed-rank.
  Effect sizes: Cohen's dz (t), rank-biserial r (Wilcoxon), discordant-pair difference (McNemar). Holm correction within each method pair.
- Comparisons are run separately at each call budget (equal LLM calls); Holm correction is applied within each method pair and budget. M1 appears at every budget as a one-call reference and M4 as a no-LLM reference.
- Fewer than 10 paired scenarios is flagged as unreliable.

## Threats to validity (state these in the paper)
Synthetic scenarios; one LLM family; LLM nondeterminism (mitigated by repetitions, temperature and seed recorded, but Gemini does not guarantee determinism); prompt sensitivity (prompts versioned and hashed); time-limited solver reference; the verifier idea is established in prior work, so the claim is the controlled comparison, not the mechanism.
