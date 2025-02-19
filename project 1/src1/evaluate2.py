from transformers import T5ForConditionalGeneration, AutoTokenizer
from evaluate import load
import json
import re

def preprocess_text(text):
    # 소문자로 변환, 구두점 제거, 중복 공백 제거
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # 구두점 제거
    text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거
    return text

def summarize(text, model, tokenizer, max_input_length=512, max_output_length=100, min_output_length=50):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length  # 입력 텍스트 최대 토큰 수 설정
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # 요약 생성
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,       # 출력 요약의 최대 토큰 수
        min_length=min_output_length,       # 출력 요약의 최소 토큰 수
        length_penalty=1.0,                 # 긴 요약 선호도 감소
        num_beams=4,                        # 빔 서치 사용
        early_stopping=True                 # 적절한 시점에서 정지
    )

    # 출력 텍스트 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def evaluate_summaries(test_data, model, tokenizer):
    rouge = load("rouge")
    bleu = load("bleu")

    references = []
    predictions = []

    for item in test_data:
        reviews_combined = " ".join(item["reviews"])
        generated_summary = summarize(reviews_combined, model, tokenizer)
        reference_summary = item["summary"]

        references.append(reference_summary)  # BLEU용 참조 요약 저장
        predictions.append(generated_summary)  # 생성된 요약 저장

        # ROUGE 업데이트 (전처리 적용)
        rouge.add(
            prediction=preprocess_text(generated_summary),
            reference=preprocess_text(reference_summary)
        )

    # ROUGE 점수 계산
    rouge_scores = rouge.compute()

    # BLEU 점수 계산
    bleu_scores = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]  # 다중 참조 형식으로 BLEU 계산
    )

    return rouge_scores, bleu_scores, predictions, references

if __name__ == "__main__":
    # 학습된 모델 불러오기
    model = T5ForConditionalGeneration.from_pretrained("../models/new_trained_model")
    tokenizer = AutoTokenizer.from_pretrained("../models/new_trained_model")

    # 테스트 데이터 로드
    test_data_path = "../data/test_final.json"  # JSON 파일 경로

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 요약 생성 및 성능 평가
    rouge_scores, bleu_scores, predictions, references = evaluate_summaries(test_data, model, tokenizer)

    # 결과 출력
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        print(f"Sample {i + 1}:")
        print(f"Generated Summary: {prediction}")
        print(f"Reference Summary: {reference}\n")

    print("ROUGE Scores:", rouge_scores)
    print("BLEU Score:", bleu_scores)
