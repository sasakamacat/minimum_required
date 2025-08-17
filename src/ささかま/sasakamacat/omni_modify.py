import os
from datasets import load_dataset, Dataset
from huggingface_hub import login

def main():
    """
    Omni-MATHデータセットをロードし、整形してHugging Face Hubにアップロードするスクリプト。
    """
    # 1. データセットのロード
    # split='test' を指定してテストデータのみをロード
    print("Loading Omni-MATH dataset...")
    dataset = load_dataset("KbsdJames/Omni-MATH", split='test')

    # 2. データの整形
    print("Formatting dataset...")
    # 'id'列を追加
    dataset = dataset.add_column("id", list(range(len(dataset))))

    # 必要な列のみを選択し、列名を変更
    dataset = dataset.select_columns(['id', 'problem', 'solution', 'answer'])
    dataset = dataset.rename_columns({
        "problem": "question",
        "solution": "thinking",
    })

    # 'thinking'列に思考プロセスを示すタグを追加
    dataset = dataset.map(
        lambda x: {
            **x,
            "thinking": f"<think>{x['thinking']}</think>\n\n\n\n"
        }
    )

    # 'content'列を追加（思考プロセスと最終回答を結合）
    def concatenate_content(examples):
        return {"content": [t + a for t, a in zip(examples["thinking"], examples["answer"])]}

    dataset = dataset.map(concatenate_content, batched=True)
    
    # 3. Hugging Face Hubへのアップロード
    print("Uploading to Hugging Face Hub...")
    try:
        # Hugging Faceにログイン（トークンが必要）
        login()

        # Hugging Face Hubにプッシュ
        dataset.push_to_hub(
            "suzakuteam/test_sasakama",
            private=False,
            commit_message="Add formatted data from Omni-MATH"
        )
        print("Successfully uploaded to suzakuteam/test_sasakama.")

    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()
