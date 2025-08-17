# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from datetime import datetime

# 非同期処理
client = AsyncOpenAI(
    timeout=300.0,
    max_retries=1,
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# 回答、reasoning、cofidenceの妥当性を図る
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""

# pydanticによる型設定
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability


# LLMが作成した駄文から評価結果をjudgeモデルを用いて出力    
async def extract_answer(args, question, correct_answer, response):
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = await client.beta.chat.completions.parse(
                model=args.judge,
                max_completion_tokens=4096, # overkill for judge
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtractedAnswer, 
            ) 
        content = response.choices[0].message.parsed
        return { 
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e: # very, very rare
        print("Error:", e)
        return None

# 評価済みでない場合、評価を行う         
async def add_judge_response(args, question, predictions):
    unique_id = question["id"]

    # 予測データにキーが存在しない場合は、処理をスキップ
    if unique_id not in predictions:
        # この警告メッセージは、どの問題がスキップされたか知る上で役立ちます
        print(f"Warning: No prediction found for question ID '{unique_id}'. Skipping.")
        return None, None

    prediction = copy.deepcopy(predictions[unique_id]) # not in-place
    question_text = question["question"]
    correct_answer = question["answer"]

    if "judge_response" in prediction: # already judged
        return unique_id, prediction
    
    response = prediction["response"]
    
    # judgeモデルに回答評価
    content = await extract_answer(args, question_text, correct_answer, response)

    if content is not None:
        prediction["judge_response"] = content # local in-place
        return unique_id, prediction
    else:
        return None, None

# openaiに一括で投げる並列処理
# 並列で何個投げるのか
async def judge_all_responses(args, questions, predictions):
    async def bound_func(question):
        async with semaphore:
            content = await add_judge_response(args, question, predictions)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results

# エラーの計算方法を決定するパラメータについて、
# エラーの計算方法
## '2' (デフォルト): L2ノルム（二乗平均平方根誤差）を使用。大きな誤差に敏感
## '1': L1ノルム（平均絶対誤差）を使用。
## 'infty' / 'infinity' / 'max': 最大誤差を使用
# 各「ビン」（区間）の目標となるサンプルサイズ。自信度をソートし、このサイズでビンに分割。
# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py

# todo：binごとのキャリブレーションを求めたい。
# 自信度ごとのキャリブレーションの値がわかる
def calib_err(confidence, correct, p='2', beta=50): 
    # 
    if len(confidence) < beta:
        return 0.0
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

# 
def dump_metrics(args, predictions, total_questions, all_questions):
    correct = []
    confidence = []
    correct_by_category = {
        "Math": [], # 41%
        "Physics": [], # 9%
        "Biology/Medicine": [], # 11%
        "Humanities/Social Science": [], # 9%
        "Computer Science/AI": [], # 10%
        "Engineering": [], # 4%
        "Chemistry": [], # 7%
        "Other": [], # 9%
    }
    for k, v in predictions.items():
        data = next(filter(lambda x: x["id"] == k, all_questions))
        if "judge_response" in v:
            judge_response = v["judge_response"]
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
            correct_by_category[data["category"]].append("yes" in judge_response["correct"])
        else:
            print(f"Missing judge response for {k}, you should rerun the judge")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100
    n = len(correct)

    # sometimes model collapses on same questions
    if len(correct) != total_questions:
        print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")


    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=100), 2)

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    print("\n*** Accuracy Per Category ***") # ここから追加
    for category, results_list in correct_by_category.items():
        num_correct = sum(results_list) # Trueの数を数える
        num_total = len(results_list)   # そのカテゴリの総質問数

        if num_total > 0:
            print(f"- {category}: {num_correct} / {num_total} ({round(num_correct / num_total * 100, 2)}%)")
        else:
            print(f"- {category}: No questions evaluated in this category.")
    print("*****************************") # ここまで追加

    results = []
    for k,v in predictions.items():
        data = next(filter(lambda x: x["id"] == k, all_questions))
        results.append({
            "id": k,
            "category": data["category"],
            "question": data["question"],
            "user_prompt": "", # TODO
            "answer_type": data["answer_type"],
            "prediction": v["judge_response"]["model_answer"],
            "gold": v["judge_response"]["correct_answer"],
            "correct": 1 if v["judge_response"]["correct"] == "yes" else 0,
            "judgement": v["judge_response"]["reasoning"],
        })

    summary = {
        "model_name": predictions[k]["model"],
        "overall_accuracy": accuracy,
        "accuracy_per_category": {
            "Math": sum(correct_by_category["Math"]) / len(correct_by_category["Math"]) * 100 if len(correct_by_category["Math"]) else None,
            "Physics": sum(correct_by_category["Physics"]) / len(correct_by_category["Physics"]) * 100 if len(correct_by_category["Physics"]) else None,
            "Biology/Medicine": sum(correct_by_category["Biology/Medicine"]) / len(correct_by_category["Biology/Medicine"]) * 100 if len(correct_by_category["Biology/Medicine"]) else None,
            "Humanities/Social Science": sum(correct_by_category["Humanities/Social Science"]) / len(correct_by_category["Humanities/Social Science"]) * 100 if len(correct_by_category["Humanities/Social Science"]) else None,
            "Computer Science/AI": sum(correct_by_category["Computer Science/AI"]) / len(correct_by_category["Computer Science/AI"]) * 100 if len(correct_by_category["Computer Science/AI"]) else None,
            "Engineering": sum(correct_by_category["Engineering"]) / len(correct_by_category["Engineering"]) * 100 if len(correct_by_category["Engineering"]) else None,
            "Chemistry": sum(correct_by_category["Chemistry"]) / len(correct_by_category["Chemistry"]) * 100 if len(correct_by_category["Chemistry"]) else None,
            "Other": sum(correct_by_category["Other"]) / len(correct_by_category["Other"]) * 100 if len(correct_by_category["Other"]) else None,
        },
        "num_questions": n,
        "timestamp": datetime.now().isoformat(),
    }

    output_folder_name = "leaderboard/" +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(output_folder_name, exist_ok=True)
    with open(output_folder_name + "/results.jsonl", "w") as f:
        f.writelines([json.dumps(l) + "\n" for l in results])
    with open(output_folder_name + "/summary.json", "w") as f:
        json.dump(summary, f, indent=4) 


def dump_metrics_by_confidence_and_category(args, predictions, total_questions, all_questions):
    """
    Calculates metrics by confidence bins and categories and prints the results to the console.
    """
    confidence_bins = {
        "0-20%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "20-40%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "40-60%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "60-80%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "80-100%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
    }

    categories = {
        "Math": [], # 41%
        "Physics": [], # 9%
        "Biology/Medicine": [], # 11%
        "Humanities/Social Science": [], # 9%
        "Computer Science/AI": [], # 10%
        "Engineering": [], # 4%
        "Chemistry": [], # 7%
        "Other": [], # 9%
    }

    for bin_name in confidence_bins:
        for category in categories:
            confidence_bins[bin_name]["categories"][category] = {"num_samples": 0, "num_correct": 0}

    # 集計処理
    for q_id, pred in predictions.items():
        if "judge_response" in pred:
            judge_response = pred["judge_response"]
            confidence_score = judge_response["confidence"]
            is_correct = ("yes" in judge_response["correct"])

            # 予測結果に対応する質問データを取得
            question_data = next((q for q in all_questions if q["id"] == q_id), None)
            if not question_data:
                continue
            category = question_data["category"]

            bin_name = None
            if 0 <= confidence_score < 20:
                bin_name = "0-20%"
            elif 20 <= confidence_score < 40:
                bin_name = "20-40%"
            elif 40 <= confidence_score < 60:
                bin_name = "40-60%"
            elif 60 <= confidence_score < 80:
                bin_name = "60-80%"
            elif 80 <= confidence_score <= 100:
                bin_name = "80-100%"

            if bin_name:
                confidence_bins[bin_name]["categories"][category]["num_samples"] += 1
                confidence_bins[bin_name]["total_in_bin"] += 1
                if is_correct:
                    confidence_bins[bin_name]["categories"][category]["num_correct"] += 1
                    confidence_bins[bin_name]["correct_in_bin"] += 1

    # コンソール出力
    for bin_name, bin_data in confidence_bins.items():
        total_samples = bin_data["total_in_bin"]

        if total_samples > 0:
            print(f"\n*** Accuracy for Confidence {bin_name} ***")
            for category in categories:
                cat_data = bin_data["categories"][category]
                num_correct = cat_data["num_correct"]
                num_total = cat_data["num_samples"]
                accuracy = round(num_correct / num_total * 100, 2) if num_total > 0 else 0.0
                
                print(f"- {category}: {num_correct} / {num_total} ({accuracy}%)")
            print("*************************************")
        else:
            print(f"\n*** No predictions in Confidence Bin {bin_name} ***")
            print("*************************************")

def dump_metrics_by_raw_subject(args, predictions, all_questions):
    """
    Calculates and prints accuracy per raw subject based on judged predictions.
    """
    correct_by_raw_subject = {}

    # raw_subjectのリストを動的に生成
    raw_subjects = sorted(list(set(q["raw_subject"] for q in all_questions if "raw_subject" in q)))

    for subject in raw_subjects:
        correct_by_raw_subject[subject] = {"num_samples": 0, "num_correct": 0}

    for q_id, pred in predictions.items():
        if "judge_response" in pred:
            judge_response = pred["judge_response"]
            is_correct = ("yes" in judge_response["correct"])
            
            question_data = next((q for q in all_questions if q["id"] == q_id), None)
            if not question_data:
                continue
            
            subject = question_data.get("raw_subject")
            if subject and subject in correct_by_raw_subject:
                correct_by_raw_subject[subject]["num_samples"] += 1
                if is_correct:
                    correct_by_raw_subject[subject]["num_correct"] += 1

    # サンプル数が多いもの事にソート
    raw_subjects = sorted(
        correct_by_raw_subject.keys(),
        key=lambda s: correct_by_raw_subject[s]["num_samples"],
        reverse=True
    )
    print("\n*** Accuracy Per Raw Subject ***")
    for subject in raw_subjects:
        results_data = correct_by_raw_subject[subject]
        num_correct = results_data["num_correct"]
        num_total = results_data["num_samples"]

        if num_total > 0:
            accuracy = round(num_correct / num_total * 100, 2)
            print(f"- {subject}: {num_correct} / {num_total} ({accuracy}%)")
        else:
            print(f"- {subject}: No questions evaluated in this raw subject.")
    print("*****************************")

def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"

    output_filepath = f"judged/hle/judged_qwen3-32B.json"   
    dataset = load_dataset(args.dataset, split="test")
    # フィルタリングしたいidのリスト
    selected_ids_list = ['66fea0c4cb66b0e85c55ee52', '67216f0abddbccdce9955e93', '66eaf9bbb082c5e6a76a49b5', '66f02cc56ea864e80ba2d3d7', '67381ce26a5242a22fe4681f', '677609438253c248a60436cb', '671d51ba8411174e64735f4d', '66f63324376699e7c6894239', '6718106f713d2e285effa910', '67d49da91dfc5429adf8e0fc', '66ef98088171e6ed71be1454', '672a857f44b1b51ebe496467', '6754baec618d187bb3362936', '6736fbf13f0517b6e8a6071e', '672556af66f8db005694c4d8', '670dc30acfd3fc87a109a91e', '6735a2d69caf928ddf5957ed', '6761a307c12ca815b1c97374', '67041871905f1ae093e7ac69', '67320e338f6d9d8c50dca222', '6718977144d804f3d0418866', '67149db7352d3701e7171a01', '66ecd3c6e8b95a8f971fb485', '673738b81bcb9388d8e3b31a', '671d70b42f937e77ea5cb0cc', '673963bfc066d38d05efee80', '671d64ecb996cf9936d1afcf', '674362ab6b7aa212dfb90595', '670c2d61886695e43e7c18b3', '673b198c2e9d1d0e28ee75ca', '6737c6ff3b2291333fbb48a4', '671f4997a2bc63fb20c63134', '672caa8a948e810916427a4c', '6716aa92d9d8d8ddd54ad7dc', '676df4158b07cdbcd761e9f2', '67387af4aa63aa9bc8096f48', '671d9c62355c956ce7de5a89', '671c6fb38668b9d1652a1645', '6724a01ff14c6dca507d5e18', '66f4a7ba439f15c2c0752479', '6705022feb89a47d12541a36', '6737309d1988146a57ffab18', '6712290fcecd80f50716f9aa', '6709986e9a83b4ea926a3036', '675d7b901ded33d59eb2c94f', '6718d2c20bcda71f53b0fe55', '67073a60c033e34a5d675f2e', '6747c521a7e2f8061effd365', '67378355cf21a88cec0f42e5', '66ea355dd1767539f1308fc3', '67178b22b7f6dd9be2a4dd68', '6706c88503718618700edfbc', '67242a2d2a37861b051e6178', '672c973499e1076f34973bfc', '66e8ea4cacb34dd5dbffa298', '670c8b10148f2a113537c8f6', '6716dc349a0dd17680df31df', '672614260019e08d7f82b2d0', '66fc006ad058f160ec5a48a1', '671aad4a17615f09eab6fe3d', '673a2f99b526858a94683ad1', '6714e9dacbabe50aa0256711', '67435ecfa59a37c4a0f5d579', '67254dd75a5d8bd3890203c6', '66fcf48f6da31bbbe3d17e72', '66ecf59741de2844089fc54b', '66ffcfa0864258b2f971a80c', '670ee20c997568ed42988705', '671d4dc4a37bbf6a4bccbd9e', '6725e8e30a7e4f593d9c716f', '66fde2de0b21f205c121aca7', '672b82cf41fc5c68b0325382', '67352e9911e5510fc618f619', '670f72aea9f4a4bebca02b4a', '6710251d73c9583c25138ff0', '66fc2559876b283872e52acc', '6725fe6b26992c47ce3a7ef5', '672bb2bbe5bc491689003908', '670f8518ffa4ea9fb0cc5fea', '6734af9217a9687889930afd', '66fbbe8cb6f8564117792b00', '672c84e4975ce618c77a5c49', '66ed58561d24f687ee9b06bb', '67191b3025b51504c14dd870', '671aee77ed3d54e87368bc9a', '66fea77b5f1bc2f5916a4171', '671abf7f136758472327f840', '67130fff5fcf1fbb43d4a687', '66f9aac10b5d5283ffc39014', '6736f46694e6cfc37bdae690', '670e92583011a5b80bfb6c60', '67019de83ec0066dfb992223', '6732d91a87eb8fc773aaf2c0', '670bcb222407af9de8866eda', '6722e21ca33cee8da62128a4', '66fd469e7a33ca69ccc69729', '67085c82e22de5e37e883351', '66ec11b230ba0debf40b1fec', '66ffaae1068d942d32104650', '6716afcdb8bcd4d88c34cd1d', '6737224020f2fc6d8d08075c', '668825f80a642802bdfeadfa', '6734346cf3950859f67019af', '66f47471bba0ce38e1ca559f', '66eb1706d3b17bc68156e3e3', '6724f8b80c7d03e5a66cce57', '66eaf456afe364e4f477b78f', '66ee81afdec0afc28a74dc93', '66e9a39f2dad0e536fb92efa', '673277a6b33b7d6a440fecdf', '6726941826b7fc6a39fbe581', '672e50c6b9281732f28aa0f3', '672368e4c08f59acd19d8353', '67364d570965d02f400bb6d2', '6729b153b7eb78fd8315f689', '67313652f659ba7b3fd1fe40', '670285bc39fbddbbfdaffdfe', '67381a8ee4ea03183132a559', '6723ec50479384d8942cca75', '67363709b1a97400f39cdc71', '66eafb96b082c5e6a76a49c0', '66fc462f9546a49018698bf6', '6715beb7b3287d3dd4db03e6', '66fc35ef3d06493836155397', '66eaf0ba5e31c19691fe73b1', '672e6368958c5d0efe9d037d', '66e9b2899cf8fcf41599246f', '670fb58f2ca6bea76e9354a7', '66fcc286eab6d512704d96af', '6709a557139851f6bbbd7a5a', '6718577ca88093a75026b186', '6718487afba38eebcbc8034b', '66ee386f26abf77702fa5a5d', '67164d0b4c922006e9e93a8d', '6702bc770065c2f50235a34e', '66eaa5414a8ccea7bfdf2df5', '671fee575a0f70920ad3bd80', '673d6d05b7ef4f8f44830921', '66f27d65a40482f6012a4006', '6742fe5ea2e78a79e46bb220', '66fb75ded83ed7a299fdd136', '671be3853529bee9e3097b92', '66ebefa090db075818df99a9', '66f8f56e204cc74a525a23b8', '66e97fd756455f341c7849c3', '66ee60c50e499ab807af71f2', '670f39dc1dcaeb830ff6231f', '66eaeaa8afe364e4f477b777', '6732f0dac65fb3c0e912b4a9', '66ea260280667dae8de4a8ec', '67190f1eee94d36fc7734311', '6724f79b262cac26e3c38360', '6705b23880154ff607a205b5', '6711e5e05e64a53ed09449fd', '677fcb0487e03828c5c19bcd', '672910239024bd67851ae002', '672e3eeb0a6bacddc98c2217', '671f1f88e6600c2d52d9fbe6', '6710be76f85b27716ff1c4ba', '66f03563a291f119cd841967', '672d4dbdf618de6356b2fa89', '67015a7f6a2b21f149f3aaba', '673e37db8d2811de2a83c135', '6736f62aafbcf9397103d22f', '672365aa71fe0b446a1eb0a2', '672e6e4a85ed85778c0fb28e', '66e95faf8451a9b41f307932', '67016793aafba5949df10067', '66fc698fd90ebe461bfd0cc4', '672bb9be168a1a30f1f92fcc', '6709c89f2d7ed3720d4b2b61', '66f8ab9c89c09a99da336b5b', '66fc4dc04293a9638d7e0f65', '66ed23aa629fe9acce7dd93d', '6723658569bb6b9a5a4499b0', '6713a2ac0223609143188d23', '66ed2cfeaa8b7a0620149264', '671f5bf4258a6646ea2f86b3', '673eb1cfadce15d9254eb2ac', '672a5d27385accabb646acfe', '672c0f1d76fec97ec7555a3a', '6736ab1af0b1f35585e14f7d', '66e70c75bbb9b1754c0869ce', '67152f87953411f24cd994f1', '672fa2180fa0aa93e4fc789a', '66ff679163f0dbcc8b9941dd', '6704409592a84f23e5b73e95', '6759a235c0c22e78a0758d86', '671ebaf92a7c16b748fd2709', '66f2e7d9384d43333482ba75', '6734d9c868c0043c3c23a8dd', '6722072ddcce66512a82d9ed', '66fe796bc12cbe6aeaa26e69', '671659f129c718f03e48a198', '6723c6ed75ffe38ab99c4e59', '673692233b82a7ef5d60532f', '67055b15222ba55ab0b88431', '67259a64572ac41765175909', '6722ac9945d2a9b39caeedd7', '66eaf77aec7a1adc61bbb13e']
    # リストをセットに変換することで、検索が高速になります
    selected_ids_set = set(selected_ids_list)

    # ラムダ関数を使ってデータセットをフィルタリング
    dataset = dataset.filter(lambda data: data['id'] in selected_ids_set)

    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    total_questions = len(all_questions)

    # output_filepath のディレクトリ部分を取得
    output_dir = os.path.dirname(output_filepath)
    # ディレクトリが存在しない場合に作成
    os.makedirs(output_dir, exist_ok=True) 
    # todo:今回カテゴリごとにばらばらに予測を行ったから貼り付ける必要あり。
    with open(f"predictions/hle/Qwen3-32B.json", "r") as f:
        predictions = json.load(f)

    # Qwen3 will be called for unjudged responses
    results_list = asyncio.run(judge_all_responses(args, all_questions, predictions))

    # judged_predictionsをここで初期化
    judged_predictions = {}

    # cache judge output
    with open(output_filepath, "w") as f:
        json.dump(results_list, f, indent=4)

    # リストを辞書に変換
    for unique_id, updated_prediction in results_list:
        if unique_id is not None:
            judged_predictions[unique_id] = updated_prediction

    dump_metrics(args, judged_predictions, total_questions, all_questions)

    dump_metrics_by_confidence_and_category(args, judged_predictions, total_questions, all_questions)

    dump_metrics_by_raw_subject(args, judged_predictions, all_questions)