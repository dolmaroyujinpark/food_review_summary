import sentencepiece as spm
import json
import os

# JSON 데이터를 텍스트 파일로 변환
def prepare_text_file(json_paths, output_file):
    with open(output_file, "w", encoding="utf-8") as output:
        for json_path in json_paths:
            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            for entry in data:
                for review in entry.get("reviews", []):
                    output.write(review + "\n")
                if "summary" in entry:
                    output.write(entry["summary"] + "\n")
    print(f"텍스트 파일 생성 완료: {output_file}")

# SentencePiece 모델 학습
def train_sentencepiece(input_file, model_prefix, vocab_size=8000):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  # 한국어와 영어 문자 모두 포함
        model_type="bpe",
        user_defined_symbols=["<NAME>"]  # 고유명사 태그 추가
    )
    print(f"SentencePiece 모델 생성 완료: {model_prefix}.model")

# 실행
json_paths = ["./data/1215_data.json", "./data/test_1.json"]  # 기존 데이터와 새로운 데이터 경로
text_file = "./data/combined_corpus.txt"  # 통합 텍스트 파일 경로
spm_model_prefix = "./data/spm"  # 모델 저장 경로

prepare_text_file(json_paths, text_file)  # 텍스트 파일 준비
train_sentencepiece(text_file, spm_model_prefix)  # SentencePiece 모델 학습