from transformers import T5ForConditionalGeneration, AutoTokenizer

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




if __name__ == "__main__":
    # 학습된 모델 불러오기
    model = T5ForConditionalGeneration.from_pretrained("../models/trained_model_1216part1")
    tokenizer = AutoTokenizer.from_pretrained("../models/trained_model_1216part1")

    # 테스트 입력
    test_review = (
        "여수 저녁쯤에 왔어융너무 맛있어보이게 잘나오네요! 사람이너무 많아서인지 모든게 셀프 이 가격대에. 바다 하면 여수가 떠오르고 삼합하면 이곳이 떠오를거같아여 맛있구 친절해서 좋았어요 분위기 너무 좋고 맛 최고 가족여행왔는데 오랜만에 맛있게 먹었어요. 항상 관광지는 실패했는데, 여기는 성공입니다. 가족끼리 낭만포차 가시려면 여기 강추합니다. 우선 여름이면 그 포차거리보단 안쪽으로 오세요. 이 여름에 진짜 포차에서 먹으면 더울 것 같더라고요. 암튼 맛도 있고 분위기도 괜찮았어요. 여수밤바다 소주 안쓰고 맛있습니다bb 잘먹었습니당 친절하시고 삼합도 넘 맛있는데 딱새우회가 진짜 미친듯이 맛있어요 밥 먹고 2차겸 안주로 가볍게 먹었는데 딱 둘의 조합이 좋았어요딱새우회는 싱싱하고 호롱은 살짝 매콤하면서 참기름? 향 같은게 돌면서 배부르지 않은 안주로 딱이었어요 리뷰 찾아보고 왔는데 맛있고 사람 많아요근데 사람 많고 가게가 넓지 않아서 입구쪽은 더워요다른 포차들도 비슷한 것 같아서 저같은 땀쟁이들은 시원한데 잘 찾아야할듯 굿! 너무 맛있어요! 다음에 또 여수오면 재방문 하고싶네요 추천! 맛있어요! 담에 여수 오면 또 올거 같아요:) 진짜 너무 맛있구 친절하고 분위기도 좋음️️️ 또 갈 거여요 진짜 맛 나요잘생긴 삼춘들 서어빙도 좋고 친절하고음식도 짱 입니당! 가장 최고 장점은 12시부터 영업해서 점심 먹을 수 있다는 것. 뿐. 바닷가 와서 기분에 즐길 수 있는 음식인 것 같아요. 뭐 미친 듯이 맛있고 그렇진 않아요. 문어는 신선한 느낌은 없었어요. 제가 잘 몰라서 그러는 것일 수도 있지만 전국서 문어 소비량 둘째 가라면 서러운 지역에서 와서 눈높이가 높을 수도 있겠네요. 리뷰가 엄청 많이 달렸길래 대단한 가게인 줄 알았는데 리뷰 쓰면 음료수 이벤트가 있었네요. 저는 솔직하게 쓰려고 음료 안 먹었지요. 저녁에 와서 분위기에 취해 먹으면 또 다를까 싶어 다시 가보려 합니다. 볶음밥은 참 맛있었어요 분위기좋고 맛도좋고 너무 좋았습니다!문어삼합 맛집소문듣고 찾아왔어요사람많은데 자리나서 앉았네요푸짐하고 맛있어요#낭만24포차 #여수낭만포차거리 #여수낭만포차 #여수낭만포차맛집 여기는 여수 올때마다 항상 방문하는곳입니다매번 너무 맛잇어요 맛있어용 최고 여수가 문어삼합이 유명하다고해서 시켜봤는데 문어랑 차돌박이랑 야채들이 잘 어울려요 :) 분위기도 이쁘고 다음에 또 오고싶은 집입니다! 작년에 여수여행때 문어삼합 왔었는데 이번 연도에 또 여기로 방문 했어요 가격은 올랐지만 정말 정말 맛있습니다. 볶음밥도 정말 맛있구요 포장도 가능하고 네이버 예약 하시면 5% 할인도 되요! 내일 가족들이랑 또 올 생각이에요 여수 여행 오시면 이곳 강추 드립니다! 내돈내산 솔직 후기 입니다! 분위기 좋고 맛도 좋아요!돌문어 차돌삼합 먹었는데, 다음에는 딱새우 회먹으러 올게요! 또갈집 이동네 포차중에 여기만 사람 많은 이유를 알겠어요 비주얼 최고에요 가격 다른인근가게와 동결분위기 최고맛은 중간 정말 맛있어요 항상 너무 맛있어요️담에도 올게여! 맛있었다 나중에 또오고싶다 회무침 문어삼합 낙지탕탕 문어숙회 문어라면 새우회 여수가면 꼭 먹어야할 메뉴가 한집에 다있어요 너무 맛있고 분위기 좋게 술 마실 수 있어요 양도 푸짐합니다 차돌삼합 두명이서 대 사이즈 시켰는데 맛있게 다먹었습니다. #낭만24포차 #여수낭만포차거리 #여수낭만포차 #여수낭만포차맛집 차돌삼합 미쳐따바다 분위기도 미쳤다소주가 술술 들어감",
    )

    # 요약 결과 출력
    print("요약 결과:", summarize(test_review, model, tokenizer, max_output_length=100, min_output_length=55))
