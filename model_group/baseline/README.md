# baseline
8/11までで完成しているベースラインです。ぜひ修正、追加してください。
GPU8枚での起動を想定しています。枚数を変化させる場合は、個人で変更してください

<コードの構成>
```bash
baseline
.
├── data
│   └── install_MATH.py
├── eval_aime
│   ├── aime_prediction.sh
│   ├── aime_script.sh
│   ├── conf
│   ├── hle_benchmark
│   │   ├── _configs.py
│   │   ├── __init__.py
│   │   ├── ollama_predictions.py
│   │   ├── openai_predictions.py
│   │   ├── __pycache__
│   │   ├── run_judge_results.py
│   │   └── vllm_predictions.py
│   ├── judge.py
│   ├── logs
│   ├── notebooks
│   └── predict.py
├── eval_hle
│   ├── conf
│   ├── hle_benchmark
│   │   ├── _configs.py
│   │   ├── __init__.py
│   │   ├── ollama_predictions.py
│   │   ├── openai_predictions.py
│   │   ├── __pycache__
│   │   ├── run_judge_results.py
│   │   └── vllm_predictions.py
│   ├── hle_prediction.sh
│   ├── hle_script.sh
│   ├── judge.py
│   ├── logs
│   ├── notebooks
│   └── predict.py
├── sequence
└── train
    ├── qwen3-32b_grpo.sh
    └── qwen3-32b_sft.sh
 ```


# 📝 リポジトリ概要

このリポジトリは、**Qwen3-32B**モデルを数学問題解決に特化させるための包括的な機械学習パイプラインを提供します。SFT（教師あり学習）とGRPO（強化学習）の2つの段階的な学習プロセスを経て、AIMEとHLEという2つの主要な数学ベンチマークでモデルを評価します。すべての処理は、SLURMジョブ管理システムと`verl`ライブラリを使用して効率的に実行されます。

---

## 📁 ディレクトリ構成

リポジトリは以下の主要なディレクトリで構成されています。

* **`train/`**: モデル学習用のSLURMスクリプトを格納します。
    * `qwen3-32b_sft.sh`: Qwen3-32Bモデルを教師あり学習（SFT）するためのスクリプトです。
    * `qwen3-32b_grpo.sh`: SFT後のモデルをGRPO（強化学習）でさらにファインチューニングするためのスクリプトです。
* **`eval_aime/`**: **AIME 2025**数学ベンチマークの評価ツールを格納します。
    * `predict.py`: モデルの予測を実行するメインスクリプトです。
    * `judge.py`: 予測結果を評価するスクリプトです。
    * `aime_script.sh`: vLLMサーバーの起動から予測、評価までを一括で実行する自動化スクリプトです。
    * `aime_prediction.sh`: 予測のみを実行するためのスクリプトです。
    * `conf/config.yaml`: Hydraで評価設定を管理するファイルです。
* **`eval_hle/`**: **HLE (High-Level Evaluation)** ベンチマークの評価ツールを格納します。構成とスクリプトは`eval_aime/`と同一です。
* **`data/`**: データセットの準備スクリプトを格納します。
    * `install_MATH.py`: MATHデータセットをダウンロードし、学習用の形式（`train.parquet`と`test.parquet`）に変換するスクリプトです。

---

## 🚀 完全な実行フローガイド

モデルの学習から評価までの一連のプロセスを3つのフェーズに分けて説明します。

### 段階 1: 環境セットアップ

モデル学習と評価を開始する前に、必要なファイル権限を設定し、データセットを準備します。

1.  **実行権限の付与**:
    すべての`.sh`スクリプトに実行権限を付与します。
    ```bash
    chmod +x ~/baseline/train/*.sh
    chmod +x ~/baseline/eval_hle/*.sh
    chmod +x ~/baseline/eval_aime/*.sh
    ```
2.  **データセットの準備**:
    `MATH`データセットをダウンロードし、学習用の形式（`train.parquet`と`test.parquet`）に変換します。
    ```bash
    python data/install_MATH.py
    ```
3.  **logを補完するディレクトリ作成**:
    ```bash
    mkdir -p ~/baseline/eval_hle/logs
    mkdir -p ~/baseline/eval_aime/logs
    ```
4.  **.shのログの書き出し位置を変更**:
    P12U007を変更してください。
    ```bash
    #SBATCH --output=/home/Competition2025/P12/P12U007/baseline/eval_hle/logs/output.out
    #SBATCH --error=/home/Competition2025/P12/P12U007/baseline/eval_hle/logs/error.out  
    ```

### 段階 2: モデル学習(動作未確認)
#### 方法1: 自動実行（未確認/方法２で実行する方がいいと思います）
ここでは、SFTとGRPOの2段階でモデルをファインチューニングします。WandBとHugging Faceの設定が必要です。

1.  **SFT（教師あり学習）**:
    ベースモデル（Qwen3-32B）を`MATH`データセットでファインチューニングします。
    * **事前準備**: `train/qwen3-32b_sft.sh`内の`HF_TOKEN`, `WANDB_ENTITY`, `WANDB_PROJECT_NAME`を設定します。
    * **実行**:
        ```bash
        cd && bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && sbatch $HOME/baseline/train/qwen3-32b_sft.sh
        ```
    * **結果**: 学習済みモデルのチェックポイントが`~/training/sft/checkpoints/`に生成されます。
2.  **GRPO（強化学習）**:
    SFTで学習したモデルを基に、GRPOでさらに性能を向上させます。
    * **事前準備**: `train/qwen3-32b_grpo.sh`内の`HF_TOKEN`, `WANDB_ENTITY`, `WANDB_PROJECT_NAME`を設定します。また、**SFTで生成されたチェックポイントパスを`qwen3-32b_grpo.sh`内の`actor_rollout_ref.model.path`に設定**する必要があります。
    * **実行**:
        ```bash
        cd && bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && sbatch $HOME/baseline/train/qwen3-32b_grpo.sh
        ```
    * **結果**: GRPO学習後のチェックポイントが`~/training/sft_grpo_001/checkpoints/`に生成されます。
#### 方法2: bashに入って実行
.shのコードを別々に実行します。

0.  **準備**:
    bashではいる
    ```bash
    bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && srun --job-name=evaluate --partition P12 --nodes=1 --nodelist osk-gpu[86] --gpus-per-node=4 --ntasks=16 --time=12:00:00 --pty bash -i
    ```
    おまじない
    ```bash
    export NCCL_SOCKET_IFNAME=enp25s0np0
    export NVTE_FUSED_ATTN=0
    #CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
    #例えば、単一GPUの場合は以下のように設定します：
    #export CUDA_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
    unset ROCR_VISIBLE_DEVICES
    ulimit -v unlimited
    ulimit -m unlimited
    ```
    conda activate
    ```bash
    export CONDA_PATH="~/conda_env"
    conda activate $CONDA_PATH
    ```
    認証系
    ```bash
    export HF_TOKEN=<Huggingfaceのトークン>
    export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
    export TRANSFORMERS_CACHE=$HF_HOME
    export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
    mkdir -p "$HF_HOME"
    echo "HF cache dir : $HF_HOME"   
    # WandBのプロジェクト設定
    export WANDB_ENTITY=<自分の名前>
    export WANDB_PROJECT_NAME="Qwen3_32B_SFT"
    export WANDB_RUN_NAME="Qwen3_32B_SFT_MATH"
    ```
   
1.  **torchrunの実行**:
2.  **チェックポイントの変換**:
3.  **モデルのアップロード(suzaku/好きな名前)**:

---

### 段階 3: 評価実行

学習済みのモデルをAIMEとHLE200問(178)のベンチマークで評価します。

#### 方法1: 自動実行（一連の動作確認済み）

この方法では、`vLLM`サーバーの起動から予測、サーバーの終了までが自動で行われます。
* **AIME評価**: `cd && bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && sbatch $HOME/baseline/eval_aime/aime_script.sh`
* **HLE評価**: `cd && bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && sbatch $HOME/baseline/eval_hle/hle_script.sh`

logs/output.outに結果が出力されます。また$HOME/judged、$HOME/leaderboard、$HOME/predictionsに結果が格納されます。

#### 方法2: bashに入って実行

vLLMサーバーを個別に管理し、予測と評価を別々に実行します。
0.  **準備**:
    ```bash
    bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && srun --job-name=evaluate --partition P12 --nodes=1 --nodelist osk-gpu[86] --gpus-per-node=4 --ntasks=16 --time=12:00:00 --pty bash -i
    ```
    ```bash
    export NCCL_SOCKET_IFNAME=enp25s0np0
    export NVTE_FUSED_ATTN=0
    #CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
    #例えば、単一GPUの場合は以下のように設定します：
    #export CUDA_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
    unset ROCR_VISIBLE_DEVICES

    ulimit -v unlimited
    ulimit -m unlimited
    ```
1.  **vLLMサーバーの起動**:
    `eval_aime/`または`eval_hle/`に移動し、`hle_script.sh`または`aime_script.sh`を実行しvllmサーバを起動
    ```bash
    nohup ./hle_script.sh > vllm.log 2>&1 &
    ```
2.  **予測の実行**:
    サーバーが起動したら、`aime_prediction.sh`または`hle_prediction.sh`を実行しタスクの推論を開始
    ```bash
    nohup ./hle_prediction.sh > prediction.log 2>&1 &
    ```
3.  **結果の評価**:
    予測が完了したら、`judge.py`を実行して評価結果を出力
    ```bash
    OPENAI_API_KEY=xxx python judge.py
    ```

---

## 🛠 設定カスタマイズの詳細

### 学習スクリプト (`.sh`)
各学習スクリプトの冒頭で以下の環境変数を設定できます。

* `export HF_TOKEN=<あなたのHugging Faceトークン>`: Hugging Face Hubへのアクセスおよびアップロードに必要です。
* `export WANDB_ENTITY="<あなたのWandB組織名>"`: WandBで学習ログを管理するための組織名です。
* `export WANDB_PROJECT_NAME="<プロジェクト名>"`: WandBでのプロジェクト名です。

### 評価設定 (`conf/config.yaml`)
`eval_aime/conf/config.yaml`または`eval_hle/conf/config.yaml`では、評価の振る舞いを詳細にカスタマイズできます。

* `model`: 評価したい学習済みモデルのパスを指定します。
* `max_completion_tokens`: モデルの応答トークン数の上限を調整します。
* `reasoning`: `true`に設定すると、モデルに思考プロセスを生成させ、CoT（Chain-of-Thought）スタイルの推論を有効にします。
* `num_workers`: 予測を並列で実行するワーカー数です。
* `max_samples`: 評価対象とするデータセットのサンプル数を指定します。

---

## 🎯 ベースラインチャート
<img width="1210" height="600" alt="image" src="https://github.com/user-attachments/assets/e5bf2573-c143-47bb-9735-d84888a5c986" />
時間の都合上、DeepMATHのGRPO部分はできていないです。