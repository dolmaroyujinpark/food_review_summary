import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Flask 앱 설정
app = Flask(__name__, static_folder="static")

# 모델 경로 (T5 모델만 유지)
T5_MODEL_PATH = "project 1/models/new_trained_model"

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# T5 모델과 토크나이저 초기화
try:
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(device)
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
    print("✅ T5 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    t5_model, t5_tokenizer = None, None

# T5 요약 함수
def t5_summarize(text, max_input_length=512, max_output_length=100, min_output_length=50):
    if not text.strip():
        return "⚠️ 입력된 텍스트가 없습니다."

    inputs = t5_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    ).to(device)

    try:
        outputs = t5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            min_length=min_output_length,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )
        summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"❌ 요약 중 오류 발생: {e}")
        return "⚠️ 요약 중 문제가 발생했습니다."

# 정적 파일 제공 엔드포인트
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# T5 요약 엔드포인트
@app.route('/summarize', methods=['POST'])
def summarize_with_t5():
    if not t5_model or not t5_tokenizer:
        return jsonify({'error': '모델이 정상적으로 로드되지 않았습니다.'}), 500

    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': '요약할 텍스트가 없습니다.'}), 400

    summary = t5_summarize(text)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)