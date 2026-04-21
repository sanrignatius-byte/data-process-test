#!/bin/bash

# Submit the repaired trial57 retrieval pipeline with explicit lane separation:
#   37  graph_only fair chunk eval
#   38  graph_only fair chunk rerank
#   37b trial57 enrich backfill
#   37bb trial57 rebuild chunk graphs after enrich
#   37c enriched fair chunk eval on rebuilt chunk graphs
#   38b enriched fair chunk rerank on rebuilt chunk graph

set -euo pipefail

cd /projects/myyyx1/data-process-test/slurm_scripts

JOB37=$(sbatch --parsable 37_chunk_corpus_eval.sh)
JOB38=$(sbatch --parsable --dependency=afterok:${JOB37} 38_chunk_graph_rerank.sh)
JOB37B=$(sbatch --parsable 37b_trial57_backfill_enrich.sh)
JOB37BB=$(sbatch --parsable --dependency=afterok:${JOB37B} 37bb_trial57_rebuild_chunk_graphs.sh)
JOB37C=$(sbatch --parsable --dependency=afterok:${JOB37BB} 37c_chunk_corpus_eval_enriched_fair.sh)
JOB38B=$(sbatch --parsable --dependency=afterok:${JOB37C} 38b_chunk_graph_rerank_enriched_fair.sh)

echo "Submitted jobs:"
echo "  37  graph_only fair eval:            ${JOB37}"
echo "  38  graph_only fair rerank:          ${JOB38}  (after ${JOB37})"
echo "  37b trial57 backfill enrich:         ${JOB37B}"
echo "  37bb rebuild chunk graphs:           ${JOB37BB} (after ${JOB37B})"
echo "  37c enriched fair eval:              ${JOB37C} (after ${JOB37BB})"
echo "  38b enriched fair rerank:            ${JOB38B} (after ${JOB37C})"
