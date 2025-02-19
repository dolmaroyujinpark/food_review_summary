import pandas as pd
import warnings
import torch
import json
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 경고 무시 설정
warnings.filterwarnings("ignore")

# 리뷰 데이터 읽기
input_file = "reviews22-1.csv"
output_file = "result_data22-1.json"

df = pd.read_csv(input_file)

# 결과 저장을 위한 리스트
result_data = []

# place name 별로 그룹화
grouped = df.groupby('Place Name')

# 각 place name 그룹에 대해 처리
for place_name, group in grouped:
    reviews = group['Review'].tolist()

    # 리뷰를 20개씩 묶어서 요약 생성
    for i in range(0, len(reviews), 30):
        batch_reviews = reviews[i:i + 30]  # 30개씩 묶기
        batch_text = ' '.join(batch_reviews)  # 리뷰 합치기

        # 결과 저장
        result_data.append({
            "restaurant_name": place_name,
            "reviews": batch_reviews,
            "summary": ""
        })
        print(f"요약 생성 완료: {place_name}, Batch {i // 30 + 1}")

# 결과를 JSON 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)

print(f"결과 저장 완료: {output_file}")
