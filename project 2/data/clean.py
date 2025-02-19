import json
import re

# reviews 리스트에서 물음표, 온점, 물결, 느낌표, 캐럿을 처리하는 함수
def clean_reviews(reviews):
    cleaned = []
    for review in reviews:
        # 물음표 연속 사용을 하나로 대체
        cleaned_review = re.sub(r'\?{2,}', '?', review)
        # 온점 연속 사용(4개 이상)을 하나로 대체
        cleaned_review = re.sub(r'\.{2,}', '.', cleaned_review)
        # 느낌표 연속 사용을 하나로 대체
        cleaned_review = re.sub(r'!{2,}', '!', cleaned_review)
        # 물결(~)과 캐럿(^)을 공백으로 대체
        cleaned_review = re.sub(r'[~^]', '', cleaned_review)
        # 공백을 정리 (여러 공백을 하나로)
        cleaned_review = re.sub(r'\s{2,}', '', cleaned_review).strip()
        cleaned.append(cleaned_review)
    return cleaned

# JSON 파일을 처리하는 함수
def process_reviews(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # JSON 파일 로드

    # 모든 항목에 대해 reviews 리스트를 수정
    for item in data:
        if 'reviews' in item:
            item['reviews'] = clean_reviews(item['reviews'])

    # 수정된 데이터를 새로운 JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 파일 경로
input_file = '1214_data.json'  # 원본 JSON 파일 경로
output_file = '1214_data.json'  # 결과를 저장할 JSON 파일 경로

# 함수 호출
process_reviews(input_file, output_file)
