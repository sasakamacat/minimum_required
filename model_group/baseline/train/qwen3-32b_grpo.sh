#!/bin/bash
#SBATCH --job-name=grpo_qwen32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00
#SBATCH --output=/home/Competition2025/P12/P12U007/baseline/train/logs/output.out
#SBATCH --error=/home/Competition2025/P12/P12U007/baseline/train/logs/error.out

################################################################################
# スクリプト名: qwen3-32b_grpo.sh
# 概要:
#   Qwen-3 32Bモデルに対し、強化学習手法GRPO（Generalized Rank-based Policy Optimization）
#   を用いたファインチューニングを行うためのSBATCHジョブスクリプト。
#
# 目的:
#   数学問題（GSM8Kデータセット）におけるモデルの推論能力を向上させる。
#   学習後、Hugging Faceフォーマットに変換し、オプションでHugging Face Hubにアップロードする。
#
# 前提条件:
#   - 環境構築が終わっていること
#   - 学習したいモデルのチェックポイントがきちんと指定されたパスに存在すること。
#   - SBATCHを修正していること
#   - Hugging Face Hubへのアップロードには、`HF_TOKEN`環境変数が設定されていること。
#
# 実行方法:
#   sbatch qwen3-32b_grpo.sh
#
# 作成者: Metokiさんを参考にさせていただきました。
# 作成日: 2025-08-10
################################################################################

# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

# Hugging Face 認証
export HF_TOKEN=<Huggingfaceのトークン>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

# エラー時に停止
set -e

# Step 1-4: 強化学習（GRPO）の実行
echo "=== Step 1-4: GRPO Training ==="


# ディレクトリ作成（パス統一）
mkdir -p ~/training/sft_grpo_001
mkdir -p ~/training/sft_grpo_001/checkpoints
cd ~/training/sft_grpo_001

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited
ulimit -m unlimited

# Ray クラスターの起動
echo "Starting Ray cluster..."
ray stop  # 既存のRayプロセスを停止
# Rayのヘッドノードを起動
# --num-cpusはノードのCPU数に合わせて調整
# --num-gpusは使用するGPUの数に合わせて調整
ray start --head --port=6379 --num-cpus=240 --num-gpus=8
echo "Ray cluster started"

# 名前は自分のものに修正してください
export WANDB_ENTITY="catnyancat"
export WANDB_PROJECT_NAME="Qwen3_32B_SFT+GRPO"
export WANDB_RUN_NAME="Qwen3_32B_SFT_MATH"

echo "Starting GRPO training..."

# GRPO学習実行
# actor_rollout_ref.model.pathを学習したいモデルに変更してください

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=128 \
 data.max_prompt_length=256 \
 data.max_response_length=1024 \
 data.dataloader_num_workers=0 \
 actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 actor_rollout_ref.actor.optim.lr=5e-7 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.default_local_dir=$HOME/training/sft_grpo_001/checkpoints \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$WANDB_PROJECT_NAME \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.total_epochs=15 2>&1 | tee verl_grpo.log

echo "GRPO training completed"

# Step 1-5: チェックポイントの変換
echo "=== Step 1-5: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_grpo_001/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT/actor \
    --target_dir $LATEST_CHECKPOINT/actor/huggingface

echo "Checkpoint conversion completed"

# Step 1-6: モデルのアップロード（オプション）
echo "=== Step 1-6: Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        Ta1k1/Qwen3-32B-SFT-GRPO \
        $LATEST_CHECKPOINT/actor/huggingface \
        --token $HF_TOKEN
    echo "Model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload Ta1k1/Qwen3-32B-SFT-GRPO $LATEST_CHECKPOINT/actor/huggingface --token YOUR_TOKEN"
fi

echo "=== GRPO Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/actor/huggingface"
