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
from dataclasses import dataclass
import os
import json
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from hle_benchmark._configs import Config
import datetime # datetimeモジュールをインポート


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def format_message(args, question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(args, question):
    messages = format_message(args, question)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            # temperature=args.temperature if "o1" not in args.model else None,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        ) 
        content = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            return None
        tokens = json.loads(response.usage.model_dump_json())
        if args.reasoning:
            tokens["reasoning_content"] = response.choices[0].message.reasoning_content
    except Exception as e:
        print("Error:", e)
        return None
    
    if content is None: # failed
        return None
    
    return question["id"], content, tokens


# 回答が10問ごとに保存
async def attempt_all(args, questions, predictions, output_filepath):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(args, question)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = []
        completed_count = 0
        
        # Process results as they complete and save every 10 items
        for result in tqdm_asyncio.as_completed(tasks):
            result_data = await result
            results.append(result_data)
            
            if result_data is not None:
                unique_id, response, usage = result_data
                predictions[unique_id] = {
                    "model": args.model,
                    "response": response,
                    "usage": usage
                }
                completed_count += 1
                print(f"Processed question ID: {unique_id}. Total completed: {completed_count}")
                
                # Save every 10 completed items
                if completed_count % 10 == 0:
                    with open(output_filepath, "w") as f:
                        json.dump(predictions, f, indent=4)
                    print(f"Saved progress: {completed_count} items completed")
    
    return results

def main(args: Config):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=30,
        api_key="fakeapikey",
    )

    assert args.num_workers > 1, "num_workers must be 2 or greater"
    dataset = load_dataset(args.dataset, split="test")
    # 画像フィルタ
    dataset = dataset.filter(lambda item: item['image'] == "")

    # カテゴリをフィルタ
    if args.category_filter:
        category_filters_lower = [c.lower() for c in args.category_filter]
        dataset = dataset.filter(lambda item: item["category"].lower() in category_filters_lower)

    # フィルタリングしたいidのリスト
    selected_ids_list = ['66fea0c4cb66b0e85c55ee52', '67216f0abddbccdce9955e93', '66eaf9bbb082c5e6a76a49b5', '66f02cc56ea864e80ba2d3d7', '67381ce26a5242a22fe4681f', '677609438253c248a60436cb', '671d51ba8411174e64735f4d', '66f63324376699e7c6894239', '6718106f713d2e285effa910', '67d49da91dfc5429adf8e0fc', '66ef98088171e6ed71be1454', '672a857f44b1b51ebe496467', '6754baec618d187bb3362936', '6736fbf13f0517b6e8a6071e', '672556af66f8db005694c4d8', '670dc30acfd3fc87a109a91e', '6735a2d69caf928ddf5957ed', '6761a307c12ca815b1c97374', '67041871905f1ae093e7ac69', '67320e338f6d9d8c50dca222', '6718977144d804f3d0418866', '67149db7352d3701e7171a01', '66ecd3c6e8b95a8f971fb485', '673738b81bcb9388d8e3b31a', '671d70b42f937e77ea5cb0cc', '673963bfc066d38d05efee80', '671d64ecb996cf9936d1afcf', '674362ab6b7aa212dfb90595', '670c2d61886695e43e7c18b3', '673b198c2e9d1d0e28ee75ca', '6737c6ff3b2291333fbb48a4', '671f4997a2bc63fb20c63134', '672caa8a948e810916427a4c', '6716aa92d9d8d8ddd54ad7dc', '676df4158b07cdbcd761e9f2', '67387af4aa63aa9bc8096f48', '671d9c62355c956ce7de5a89', '671c6fb38668b9d1652a1645', '6724a01ff14c6dca507d5e18', '66f4a7ba439f15c2c0752479', '6705022feb89a47d12541a36', '6737309d1988146a57ffab18', '6712290fcecd80f50716f9aa', '6709986e9a83b4ea926a3036', '675d7b901ded33d59eb2c94f', '6718d2c20bcda71f53b0fe55', '67073a60c033e34a5d675f2e', '6747c521a7e2f8061effd365', '67378355cf21a88cec0f42e5', '66ea355dd1767539f1308fc3', '67178b22b7f6dd9be2a4dd68', '6706c88503718618700edfbc', '67242a2d2a37861b051e6178', '672c973499e1076f34973bfc', '66e8ea4cacb34dd5dbffa298', '670c8b10148f2a113537c8f6', '6716dc349a0dd17680df31df', '672614260019e08d7f82b2d0', '66fc006ad058f160ec5a48a1', '671aad4a17615f09eab6fe3d', '673a2f99b526858a94683ad1', '6714e9dacbabe50aa0256711', '67435ecfa59a37c4a0f5d579', '67254dd75a5d8bd3890203c6', '66fcf48f6da31bbbe3d17e72', '66ecf59741de2844089fc54b', '66ffcfa0864258b2f971a80c', '670ee20c997568ed42988705', '671d4dc4a37bbf6a4bccbd9e', '6725e8e30a7e4f593d9c716f', '66fde2de0b21f205c121aca7', '672b82cf41fc5c68b0325382', '67352e9911e5510fc618f619', '670f72aea9f4a4bebca02b4a', '6710251d73c9583c25138ff0', '66fc2559876b283872e52acc', '6725fe6b26992c47ce3a7ef5', '672bb2bbe5bc491689003908', '670f8518ffa4ea9fb0cc5fea', '6734af9217a9687889930afd', '66fbbe8cb6f8564117792b00', '672c84e4975ce618c77a5c49', '66ed58561d24f687ee9b06bb', '67191b3025b51504c14dd870', '671aee77ed3d54e87368bc9a', '66fea77b5f1bc2f5916a4171', '671abf7f136758472327f840', '67130fff5fcf1fbb43d4a687', '66f9aac10b5d5283ffc39014', '6736f46694e6cfc37bdae690', '670e92583011a5b80bfb6c60', '67019de83ec0066dfb992223', '6732d91a87eb8fc773aaf2c0', '670bcb222407af9de8866eda', '6722e21ca33cee8da62128a4', '66fd469e7a33ca69ccc69729', '67085c82e22de5e37e883351', '66ec11b230ba0debf40b1fec', '66ffaae1068d942d32104650', '6716afcdb8bcd4d88c34cd1d', '6737224020f2fc6d8d08075c', '668825f80a642802bdfeadfa', '6734346cf3950859f67019af', '66f47471bba0ce38e1ca559f', '66eb1706d3b17bc68156e3e3', '6724f8b80c7d03e5a66cce57', '66eaf456afe364e4f477b78f', '66ee81afdec0afc28a74dc93', '66e9a39f2dad0e536fb92efa', '673277a6b33b7d6a440fecdf', '6726941826b7fc6a39fbe581', '672e50c6b9281732f28aa0f3', '672368e4c08f59acd19d8353', '67364d570965d02f400bb6d2', '6729b153b7eb78fd8315f689', '67313652f659ba7b3fd1fe40', '670285bc39fbddbbfdaffdfe', '67381a8ee4ea03183132a559', '6723ec50479384d8942cca75', '67363709b1a97400f39cdc71', '66eafb96b082c5e6a76a49c0', '66fc462f9546a49018698bf6', '6715beb7b3287d3dd4db03e6', '66fc35ef3d06493836155397', '66eaf0ba5e31c19691fe73b1', '672e6368958c5d0efe9d037d', '66e9b2899cf8fcf41599246f', '670fb58f2ca6bea76e9354a7', '66fcc286eab6d512704d96af', '6709a557139851f6bbbd7a5a', '6718577ca88093a75026b186', '6718487afba38eebcbc8034b', '66ee386f26abf77702fa5a5d', '67164d0b4c922006e9e93a8d', '6702bc770065c2f50235a34e', '66eaa5414a8ccea7bfdf2df5', '671fee575a0f70920ad3bd80', '673d6d05b7ef4f8f44830921', '66f27d65a40482f6012a4006', '6742fe5ea2e78a79e46bb220', '66fb75ded83ed7a299fdd136', '671be3853529bee9e3097b92', '66ebefa090db075818df99a9', '66f8f56e204cc74a525a23b8', '66e97fd756455f341c7849c3', '66ee60c50e499ab807af71f2', '670f39dc1dcaeb830ff6231f', '66eaeaa8afe364e4f477b777', '6732f0dac65fb3c0e912b4a9', '66ea260280667dae8de4a8ec', '67190f1eee94d36fc7734311', '6724f79b262cac26e3c38360', '6705b23880154ff607a205b5', '6711e5e05e64a53ed09449fd', '677fcb0487e03828c5c19bcd', '672910239024bd67851ae002', '672e3eeb0a6bacddc98c2217', '671f1f88e6600c2d52d9fbe6', '6710be76f85b27716ff1c4ba', '66f03563a291f119cd841967', '672d4dbdf618de6356b2fa89', '67015a7f6a2b21f149f3aaba', '673e37db8d2811de2a83c135', '6736f62aafbcf9397103d22f', '672365aa71fe0b446a1eb0a2', '672e6e4a85ed85778c0fb28e', '66e95faf8451a9b41f307932', '67016793aafba5949df10067', '66fc698fd90ebe461bfd0cc4', '672bb9be168a1a30f1f92fcc', '6709c89f2d7ed3720d4b2b61', '66f8ab9c89c09a99da336b5b', '66fc4dc04293a9638d7e0f65', '66ed23aa629fe9acce7dd93d', '6723658569bb6b9a5a4499b0', '6713a2ac0223609143188d23', '66ed2cfeaa8b7a0620149264', '671f5bf4258a6646ea2f86b3', '673eb1cfadce15d9254eb2ac', '672a5d27385accabb646acfe', '672c0f1d76fec97ec7555a3a', '6736ab1af0b1f35585e14f7d', '66e70c75bbb9b1754c0869ce', '67152f87953411f24cd994f1', '672fa2180fa0aa93e4fc789a', '66ff679163f0dbcc8b9941dd', '6704409592a84f23e5b73e95', '6759a235c0c22e78a0758d86', '671ebaf92a7c16b748fd2709', '66f2e7d9384d43333482ba75', '6734d9c868c0043c3c23a8dd', '6722072ddcce66512a82d9ed', '66fe796bc12cbe6aeaa26e69', '671659f129c718f03e48a198', '6723c6ed75ffe38ab99c4e59', '673692233b82a7ef5d60532f', '67055b15222ba55ab0b88431', '67259a64572ac41765175909', '6722ac9945d2a9b39caeedd7', '66eaf77aec7a1adc61bbb13e']
    # リストをセットに変換することで、検索が高速になります
    selected_ids_set = set(selected_ids_list)

    # ラムダ関数を使ってデータセットをフィルタリング
    dataset = dataset.filter(lambda data: data['id'] in selected_ids_set)

    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        all_questions = all_questions[:args.max_samples]

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_filepath = f"predictions/hle/Qwen3-32B.json"
    # もしディレクトリがない場合は作成
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # 同じ問題は解かないためのフィルタリング機能
    # 設定したpredictionファイルについて被りがないように問題をフィルタ
    if args.duplicate_filter:
        # load only questions without responses
        if os.path.exists(args.duplicate_file):
            with open(args.duplicate_file, "r") as f:
                predictions = json.load(f)
            unsolved_questions = [q for q in all_questions if q["id"] not in predictions]
        else:
            print("args.duplicate_fileの設定が間違っている")

    while unsolved_questions:
        print(f"Attempting to solve {len(unsolved_questions)} remaining questions...")
        
        results = asyncio.run(attempt_all(args, unsolved_questions, predictions, output_filepath))
        
        for result in results:
            if result is not None:
                unique_id, response, usage = result
                predictions[unique_id] = {
                    "model": args.model,
                    "response": response,
                    "usage": usage,
                }
        
        with open(output_filepath, "w") as f:
            json.dump(predictions, f, indent=4)
            
        print(f"Current progress saved: {len(predictions)} / {len(all_questions)} questions solved.")
        
        unsolved_questions = [q for q in all_questions if q["id"] not in predictions]

    print(f"All {len(all_questions)} questions have been solved and saved.")


    # Final save to ensure all data is persisted
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Final save completed: {len([r for r in results if r is not None])} total items processed")
