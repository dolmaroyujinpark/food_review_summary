import json
import re

# 초성 제거 정규 표현식
def remove_chosung(text):
    return re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)

# 이모지 제거 정규 표현식
def remove_emojis(text):
    emoji_pattern = re.compile(
        # 감정, 사람 관련 이모지
        "[\U0001F600-\U0001F64F"  
        # 기호, 자연
        "\U0001F300-\U0001F5FF"  
        # 교통수단, 기계
        "\U0001F680-\U0001F6FF"
        # 기타 기호 및 기타 카테고리
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"  # 기호
        "\U00002700-\U000027BF"  # 기호들
        # 별표 등 특정 기호
        "\U00002B50"             # ⭐ 별표
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

    # 이모지를 빈 문자열로 대체
    return emoji_pattern.sub(r'', text)

# reviews 처리 함수: 리스트 형태 유지
def process_reviews(reviews):
    if isinstance(reviews, list):
        # 리스트의 각 항목 처리
        return [
            remove_chosung(remove_emojis(review)) if isinstance(review, str) else review
            for review in reviews
        ]
    return reviews

# JSON 파일 읽기
with open('test_2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# reviews 항목 처리
for restaurant in data:
    if 'reviews' in restaurant:
        restaurant['reviews'] = process_reviews(restaurant['reviews'])

# 수정된 데이터를 새로운 파일로 저장
with open('test_2.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("completed")
