#!/bin/bash
# ============================================================
# M4 Workflow - 提交所有任务（带依赖）
#
# 用法:
#   ./slurm_scripts/submit_all.sh           # 提交全部3个任务
#   ./slurm_scripts/submit_all.sh --skip-download  # 跳过下载
#   ./slurm_scripts/submit_all.sh --parse-only     # 只运行解析
# ============================================================

cd /projects/myyyx1/data-process-test

# 确保logs目录存在
mkdir -p logs

SKIP_DOWNLOAD=false
PARSE_ONLY=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --parse-only)
            PARSE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "M4 Workflow Submission"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Parse only: $PARSE_ONLY"
echo "=========================================="

if [ "$SKIP_DOWNLOAD" = false ]; then
    # Step 1: 提交下载任务
    JOB1=$(sbatch --parsable slurm_scripts/01_fetch_references.sh)
    echo "Submitted fetch_references: Job $JOB1"

    # Step 2: 提交解析任务（依赖下载完成）
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_scripts/02_parse_pdfs.sh)
    echo "Submitted parse_pdfs: Job $JOB2 (depends on $JOB1)"
else
    # 跳过下载，直接提交解析
    JOB2=$(sbatch --parsable slurm_scripts/02_parse_pdfs.sh)
    echo "Submitted parse_pdfs: Job $JOB2"
fi

if [ "$PARSE_ONLY" = false ]; then
    # Step 3: 提交Query生成任务（依赖解析完成）
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm_scripts/03_generate_m4_queries.sh)
    echo "Submitted m4_query_gen: Job $JOB3 (depends on $JOB2)"
fi

echo ""
echo "=========================================="
echo "All jobs submitted. Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/fetch_refs_*.out"
echo "  tail -f logs/parse_*.out"
echo "  tail -f logs/m4_query_*.out"
echo "=========================================="
