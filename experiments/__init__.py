"""One-shot experiments / pilots / ablations — Phase 7 of the 2026-04-22 decoupling plan.

This package is a placeholder for **scripts that are not part of the production
pipeline**: pilots, ablations, dated runs, and other one-off deliverables.

Files **scheduled to move here from ``scripts/`` in round 2** (pending sync of
``slurm_scripts/``)::

    build_trial57_enrich_subset.py          (referenced by slurm/37b_*.sh)
    build_crossdoc_gold57.py                (referenced by slurm/40_*.sh + 2 scripts)
    build_method_c_feasibility_candidates.py (referenced by slurm/23_*.sh)
    method_c_auto_followup.py               (referenced by slurm/11_*.sh)
    run_exp_a_difficulty.py                 (standalone)
    run_exp_c_qa_triangle.py                (standalone)
    run_m2_classic_eval.py                  (referenced by run_m2_classic_eval_oneclick.sh)
    pilot_method_c.py                       (referenced by slurm/10_*.sh + 2 scripts)

Decision rule for what lives here:

    File name contains a date, experiment code (``exp_a``/``trial57``), ``pilot``,
    ``gold57``, or ``feasibility`` → belongs in ``experiments/``.

**DO NOT physically move these in round 1** — doing so without simultaneously
updating the slurm scripts would break cluster jobs. Round 2 will handle the
move + slurm path sync as a single atomic change.
"""
