# 🍽️ 음식점 리뷰 요약기 📚✨

이 저장소는 음식점 리뷰를 요약하는 **Fine-tuned KoT5 모델** 기반의 텍스트 요약 시스템을 포함하고 있습니다.  
최종적으로 **웹 애플리케이션(`app.py`)에서는 Fine-tuned KoT5 모델만 사용됩니다.**  
(📌 참고: Sparse Transformer 모델은 `Project 2`에 포함되어 있으나, 현재 웹에서는 사용되지 않습니다.)

---

## **📌 프로젝트 개요**
이 프로젝트는 **두 개의 요약 모델을 포함**하지만, 최종적으로 **Fine-tuned KoT5 모델**을 사용합니다.

1. **✅ Project 1 (Fine-tuned KoT5 기반 요약 모델)**
   - KoT5 모델을 파인튜닝하여 텍스트 요약을 수행합니다.
   - **최종적으로 웹 애플리케이션(`app.py`)에서 사용되는 모델**입니다.

2. **📌 Project 2 (Sparse Transformer, 참고용)**
   - Transformer 구조를 직접 구현한 Sparse Transformer 모델입니다.
   - **현재 웹 애플리케이션에서는 사용되지 않으며, 참고용으로 유지됩니다.**

---

## **📌 파일 구조**

### **📂 Project 1 (웹 애플리케이션에서 최종 사용)**
project 1/ │── data/ │ ├── test_final.json # 테스트용 데이터셋 │── models/ │ ├── new_trained_model/ # Fine-tuned KoT5 모델 │── src1/ │ ├── dataset.py # 데이터셋 로드 및 처리 코드 │ ├── preprocess.py # 데이터 전처리 스크립트 │ ├── train.py # KoT5 모델 학습 코드 │ ├── evaluate1.py # 모델 평가 스크립트 │ ├── utils.py # 유틸리티 함수


### **📂 Project 2 (참고용, 웹에서는 사용되지 않음)**
project 2/ │── data/ │ ├── 1215_data.json # 학습 데이터셋 (Sparse Transformer) │── src2/ │ ├── main.py # Sparse Transformer 모델 학습 코드 │ ├── generate.py # 요약 결과 생성 │ ├── prepare_spm.py # 전처리 및 토큰화 │── README.md # Project 2 설명 문서 │── requirements.txt # 필요한 라이브러리 목록


### **📂 웹 애플리케이션**
static/ │── index.html # 웹 애플리케이션 프론트엔드

app.py # 웹 애플리케이션 백엔드 (최종적으로 Fine-tuned KoT5 모델을 사용)


---

## **📌 실행 방법**

### **1️⃣ 필수 라이브러리 설치**
```bash
pip install -r requirements.txt
2️⃣ Project 1 - KoT5 모델 실행 (웹에서 사용됨)
📌 모델 학습
python project 1/src1/train.py
📌 모델 평가
python project 1/src1/evaluate1.py
3️⃣ (참고용) Project 2 - Sparse Transformer 실행 (웹에서는 사용되지 않음)
python project 2/src2/main.py
4️⃣ 웹 애플리케이션 실행
python app.py
📌 브라우저에서 http://127.0.0.1:5000에 접속하면 Fine-tuned KoT5 모델을 사용한 요약 결과를 확인할 수 있습니다.

