#!/bin/bash
#SBATCH --job-name=AIME_evaluate
#SBATCH --partition=P12
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=6:00:00
#SBATCH --output=/home/Competition2025/P12/P12U007/baseline/eval_aime/logs/output.out
#SBATCH --error=/home/Competition2025/P12/P12U007/baseline/eval_aime/logs/error.out
#SBATCH --export=OPENAI_API_KEY=<APIKEY>
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

################################################################################
# スクリプト名: aime_script.sh
# 概要:
#   このスクリプトは、vLLMサーバーとHugging Faceモデルを使用して、
#   AIMEベンチマークでQwen3-32Bモデルを評価するためのSLURMジョブスクリプトです。
#
# 実行の流れ:
# 1. ログ記録関数を定義し、ジョブ情報を出力します。
# 2. 必要なソフトウェアモジュール（CUDA, Condaなど）をロードし、conda環境をアクティベートします。
# 3. Hugging Faceの認証情報とGPU設定（NCCL, CUDA_VISIBLE_DEVICESなど）を環境変数に設定します。
# 4. GPUの利用状況を監視するためのバックグラウンドプロセスを開始します。
# 5. 予測結果と評価結果を保存するためのディレクトリを作成します。
# 6. vLLMサーバーのキャッシュをクリアし、vLLMサーバーをバックグラウンドで起動します。
#    --tensor-parallel-sizeには、SLURMから取得したGPU数を使用します。
# 7. サーバーが応答するまでヘルスチェックを繰り返して待機します。
# 8. vLLMサーバーのモデル一覧を取得し、起動を確認します。
# 9. 予測スクリプト（predict.py）と評価スクリプト（judge.py）をバックグラウンドで実行します。
# 10. ジョブの終了時に、vLLMサーバーとGPU監視プロセスをクリーンアップします。
#
# 注意点:
#   - vLLMモデルパス: vllm serveコマンドの引数は、評価対象のモデルのパスに正確に一致させる必要があります。
#   - logパス: 出力ログ（--output, --error, vllm.logなど）のパスが、意図したディレクトリ構造と一致していることを確認してください。
#   - プロセス管理: predict.pyとjudge.pyをバックグラウンドで実行していますが、これらのプロセスが完了する前にスクリプトが終了しないよう、適切にwaitコマンドを配置する必要があります。
#
################################################################################

GPU_NUM=$SLURM_GPUS_PER_NODE

#--- log用 --------------------------------------------------------
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') [${1^^}] ${*:2}"
}
log INFO "JOB開始: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}"


# Hugging Face 認証
export HF_TOKEN=<APIKEY>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
unset ROCR_VISIBLE_DEVICES

ulimit -v unlimited
ulimit -m unlimited

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 60 > ${HOME}/baseline/eval_aime/logs/nvidia-smi.log &
pid_nvsmi=$!

#--- 必要なディレクトリを作成 -----------------------------------------
mkdir -p predictions
mkdir -p judged

#--- Initialize errorの回避 -------------------------------------------------------
rm -rf ~/.cache/vllm

#--- vLLM 起動（8GPU）----------------------------------------------
# tensor-parallel-sizeについてはmulti headsを割り切れる数に指定する必要あり
# どこでモデルのmulti headsを見れるかの手法はこちら
# 
vllm serve ${HOME}/model/Qwen3-32B \
  --tensor-parallel-size $GPU_NUM \
  --reasoning-parser deepseek_r1 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.80 \
  > ${HOME}/baseline/eval_aime/logs/vllm.log &
# $!を環境変数に保存
# 片付けの時に使う
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY" 

# モデル一覧を取得して確認
models=$(curl -s http://localhost:8000/v1/models)
echo "$models"

# hydraエラー回避
export PYTHONPATH=$HOME/.conda/envs/llmbench/lib/python3.12/site-packages:$PYTHONPATH
#--- 推論 -----------------------------------------------------------
python ${HOME}/baseline/eval_aime/predict.py 
echo "Predict.py completed."

# predict.pyが完了した後にjudge.pyを実行します。
python ${HOME}/baseline/eval_aime/judge.py 
echo "Judge.py completed."

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait
