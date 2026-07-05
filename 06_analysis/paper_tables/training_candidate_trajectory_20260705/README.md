# Step-Indexed Training Trajectory

- Created: 2026-07-05T13:12:08
- Source: `06_analysis\paper_tables\round2_transfer\training`
- Candidate evaluations: 40
- Final training step: 48000
- Generations: 5
- Best score: 126.1648
- Best candidate: generation 4, candidate 5

Interpretation: the selected policy was produced by a CEM-style shared-parameter policy search.
The x-axis is cumulative training environment steps, computed from candidate evaluations, episode length, episodes per seed, and training seeds.
Each plotted point is still one full candidate-policy episode evaluation, not a per-gradient-update reward sample.
Use these plots as step-indexed policy-search trace evidence only; do not describe them as a theoretical convergence proof.

Generated files:
- `candidate_evaluation_history.csv`
- `elite_evaluation_history.csv`
- `generation_best_history.csv`
- Final best-so-far score: 126.1648
