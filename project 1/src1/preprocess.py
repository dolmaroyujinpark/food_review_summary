import json
from transformers import AutoTokenizer


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(data, tokenizer, max_input_length=512, max_target_length=128):
    processed_data = []
    for item in data:
        review_text = " ".join(item["reviews"])
        summary = item["summary"]

        # 빈 텍스트 확인
        if not review_text.strip() or not summary.strip():
            continue

        inputs = tokenizer(
            review_text,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = tokenizer(
            summary,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        processed_data.append({
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        })
    return processed_data



if __name__ == "__main__":
    # KoT5 모델 불러오기
    model_name = "psyche/KoT5-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 데이터 로드 및 전처리
    data = load_data("../data/1216_part1.json")
    processed_data = preprocess_data(data, tokenizer)

    # 결과 출력
    print("Preprocessed data sample:", processed_data[:1])
