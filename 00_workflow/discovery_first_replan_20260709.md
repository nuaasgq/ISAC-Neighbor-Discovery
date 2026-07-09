# Discovery-First Replan for ISAC-Assisted MARL Neighbor Discovery

Date: 2026-07-09

## Reason for Replanning

The previous experiment line over-emphasized collision count and CPD
(`collisions per discovery`) when judging MARL variants. Collision is a protocol
failure mechanism and an overhead diagnostic, but it is not the scientific
objective of narrow-beam neighbor discovery. A method with fewer collisions is
not necessarily better if it discovers fewer neighbors or builds a poorer
topology.

The paper evidence line is therefore reset to a discovery-first hierarchy.

## Metric Hierarchy

### Primary metrics

These metrics decide whether a method is useful for the paper.

1. Neighbor discovery rate under a fixed slot budget.
2. Average discovery delay and tail discovery delay, especially p95/p99.
3. Cumulative discovered links versus slot.
4. Topology quality after discovery, including discovered edges, LCC ratio,
   isolated-node ratio, and algebraic connectivity.
5. Transfer performance from small-scale training to larger swarms and other
   beam widths.

### Secondary efficiency metrics

These metrics support the mechanism explanation, but cannot replace the primary
metrics.

1. Empty-scan ratio.
2. Discoveries per 1000 scan actions.
3. Discoveries per joule or per normalized access cost.
4. Table-exchange overhead if enabled.

### Diagnostic metrics only

These should not be used as the main claim.

1. Collision count.
2. Collisions per discovery / CPD.
3. Collision-penalized discovery rate.

Collision is only useful when two methods have comparable discovery and topology
performance. Then lower collision means lower invalid access, lower energy
waste, and lower channel contention. It cannot by itself prove better neighbor
discovery.

## Baseline and Comparison Reset

The Wang2025-style comparison should be evaluated mainly by discovery time and
discovery rate, because Wang's paper models a conflict mechanism but does not
use collision rate or CPD as the core evaluation metric. Their main result
language centers on sensing accuracy, sensing success probability, and consumed
slot number for neighbor discovery.

Required comparison rows:

1. Uniform random blind search.
2. Wang2025 ISAC without collaboration.
3. Wang2025 communication table exchange.
4. Wang2025 ISAC sensing-table exchange.
5. Our strongest rule-based ISAC link-layer protocol.
6. MARL without ISAC.
7. MARL with TX-coupled ISAC.
8. MARL with TX-coupled ISAC plus any proposed network/policy innovation.

Standalone SENSE remains disabled for our MARL. ISAC feedback is generated as
TX-coupled piggyback sensing from the selected transmit beam.

## Immediate Work Packages

### WP1: Clean the claim and figure inventory

Goal: remove misleading evidence pressure from CPD/collision plots.

Tasks:

1. Audit current result reports and figure manifests.
2. Mark CPD, collision count, and collision-penalized discovery figures as
   diagnostic-only.
3. Promote discovery-rate, delay, cumulative-link, topology, and empty-scan
   figures.
4. Build a new paper figure shortlist with no fewer than these groups:
   discovery rate, cumulative discovery, delay, topology, transfer, sensing
   quality/empty scan, and training convergence.

Deliverable:

- `06_analysis/discovery_first_metric_audit_20260709.md`
- updated figure/table index for manuscript use

### WP2: Recompute Wang-style comparison under primary metrics

Goal: produce a fair Wang-vs-ours table that answers the correct question.

Tasks:

1. Reuse the existing Wang-style single-RF environment.
2. Evaluate N = 10, 20, 30, 40, 50, slot budget = 200, RF = 1.
3. Report discovery rate, average consumed slot number / completion slot,
   cumulative discoveries versus slot, delay, discovered edges, and topology
   quality.
4. Keep collision only as a final diagnostic column.
5. Highlight whether any MARL row beats Wang rows on primary metrics; if not,
   state the boundary honestly.

Deliverable:

- discovery-first aggregate table
- cumulative discovery curves
- Wang-style main comparison figure set

### WP3: Redesign MARL reward and observation around discovery quality

Goal: stop optimizing a proxy that does not match the paper claim.

Reward should prioritize:

1. new direct discoveries;
2. early discoveries, using time-discounted discovery reward;
3. topology improvement, such as LCC or isolated-node reduction;
4. useful ISAC-guided beam reuse;
5. moderate access/energy cost.

Collision should be treated as a failed-access cost or channel-overhead cost,
not a dominant objective. Heavy collision penalties can make the policy passive
and hurt discovery.

Observation should include:

1. own pose and beam orientation state;
2. local sensing table summary;
3. local neighbor table summary;
4. recent selected beam/mode history;
5. local discovery progress;
6. optional density/contention estimate derived from local observations only.

Action space stays simple:

1. mode: TX / RX / IDLE;
2. transmit or receive beam;
3. optional probabilistic access gate.

No standalone SENSE action. No explicit backoff action in the first redesign.

Deliverable:

- revised MARL environment contract
- reward-ablation plan
- training curves with step-level reward and episode-level discovery metrics

### WP4: Train small, transfer large

Goal: satisfy the small-to-large scalability claim.

Training:

1. Train on N = 10.
2. Use 300 slots per episode.
3. Train at a reference beam width, initially 10 degrees.
4. Keep RF = 1.
5. Use stochastic decentralized deployment, not deterministic argmax, unless
   deterministic performance is explicitly calibrated.

Testing:

1. Transfer to N = 20, 30, 50, 100.
2. Transfer to beam widths 3, 5, 10, 15 degrees; skip 30 degrees unless needed
   as a loose-beam appendix.
3. Include both equal-area scaling and fixed-area density stress when N = 100.
4. Use Wang-style 200-slot tests for direct literature comparison and a longer
   slot budget only as robustness evidence.

Deliverable:

- transfer matrix with discovery-first metrics
- trained-model scalability plots

### WP5: Decide the paper route after evidence

Decision rule:

1. If MARL beats or matches Wang-style ISAC tables on discovery rate/delay while
   preserving topology quality, the main method can be MARL+ISAC.
2. If MARL clearly benefits from ISAC but still loses to the strongest rule
   baseline, the paper should position MARL as a learned extension and use the
   strongest rule-based ISAC protocol as the main currently validated method.
3. If neither learned nor rule-based methods beat Wang on primary metrics, stop
   paper drafting and return to mechanism design.

Manuscript claims must be limited by this decision rule.

## Stopped or Demoted Work

1. Do not optimize or select methods primarily by CPD.
2. Do not claim collision reduction as the central contribution.
3. Do not use collision-penalized discovery as the main abstract/result number.
4. Do not add standalone SENSE to MARL.
5. Do not add learned backoff until the basic TX/RX/IDLE discovery policy is
   strong on primary metrics.
6. Do not run broad 3000-slot sweeps before the Wang-style 200-slot evidence is
   corrected.

## Success Criteria for the Next Round

The next round is successful only if it produces:

1. a corrected metric audit;
2. a Wang-style discovery-first comparison table;
3. cumulative discovery curves;
4. step-level MARL training reward curves;
5. a clear yes/no answer on whether the current MARL method supports the main
   paper claim;
6. a revised claim boundary that does not rely on collision/CPD as the main
   evidence.

