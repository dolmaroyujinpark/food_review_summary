### **README**

---

## **프로젝트 개요**  
이 저장소는 두 개의 텍스트 요약 모델 프로젝트를 포함하고 있습니다.  

1. **Project 1**:  
   - **KoT5 모델**을 활용해 파인튜닝한 텍스트 요약 모델입니다.  
   - 데이터 전처리, 학습, 평가를 위한 코드가 포함되어 있습니다.  

2. **Project 2**:  
   - **Sparse Transformer**를 활용해 직접 구현한 텍스트 요약 모델입니다.  
   - 데이터 처리, 학습 코드, 그리고 결과를 테스트하는 파일이 포함되어 있습니다.  

또한 두 가지 모델을 비교할 수 있는 **웹 애플리케이션**이 포함되어 있습니다 (`app.py`).  

---

## **파일 구조**  

### **Project 1**  
- **data/**  
  - `1217_data.json` : 학습용 데이터셋.  
  - `test_final.json` : 테스트용 데이터셋.  

- **models/**  
  - `new_trained_model/` : 새롭게 학습된 최종 모델 파일이 저장된 디렉토리.  
  - `trained_model_1216part1/` : 이전 학습된 모델 파일.  

- **src1/**  
  - `dataset.py` : 데이터셋 로드 및 처리 코드.  
  - `preprocess.py` : 데이터 전처리 스크립트.  
  - `train.py` : KoT5 모델 학습 코드.  
  - `evaluate1.py`, `evaluate2.py` : 요약 모델 성능을 평가하는 스크립트.  
  - `utils.py` : 유틸리티 함수 모음.  

---

### **Project 2**  
- **data/**  
  - Project 2에서 사용할 데이터셋.  

- **src2/**  
  - `__init__.py` : 패키지 초기화 파일.  
  - `generate.py` : 요약 결과를 생성 및 테스트하는 스크립트.  
  - `generate_input_ver.py` : 입력 데이터를 활용한 요약 결과 생성.  
  - `main.py` : Sparse Transformer 모델 학습 및 실행 코드.  
  - `prepare_spm.py` : 토큰화 및 전처리 관련 스크립트.  
  - `README.md` : 프로젝트 2 설명 문서.  
  - `requirements.txt` : 필요한 라이브러리 목록.  
  - `transformer_weights.pth` : 학습된 Sparse Transformer 모델 가중치 파일.  

- **static/**  
  - `index.html` : 웹 애플리케이션용 HTML 파일.  

- **app.py**  
  - 웹 애플리케이션 실행 파일로, **Project 1**과 **Project 2**의 모델을 비교하여 요약문을 생성합니다.  

---

## **실행 방법**

### 1. **필수 라이브러리 설치**  
두 프로젝트의 필수 라이브러리를 설치합니다.  
```bash
pip install -r requirements.txt
```

### 2. **Project 1 - KoT5 모델 실행**  
#### **학습**  
KoT5 모델 학습을 실행하려면 `src1/train.py` 파일을 사용합니다.  
```bash
python src1/train.py
```

#### **평가**  
`evaluate1.py` 또는 `evaluate2.py`를 실행하여 모델 성능을 테스트합니다.  
```bash
python src1/evaluate1.py
python src1/evaluate2.py
```

---

### 3. **Project 2 - Sparse Transformer 실행**  
#### **학습**  
Sparse Transformer 모델을 학습시키려면 `main.py`를 실행합니다.  
```bash
python src2/main.py
```

#### **요약 결과 생성**  
`generate.py`를 사용하여 요약 결과를 확인합니다.  
```bash
python src2/generate.py
```

---

### 4. **웹 애플리케이션 실행 (모델 비교)**  
`app.py`를 실행하여 웹 페이지에서 두 모델을 비교할 수 있습니다.  
```bash
python app.py
```

웹 브라우저에서 `http://127.0.0.1:5000`에 접속하면 두 모델의 요약 결과를 실시간으로 비교할 수 있습니다.  

---

## **기타 참고사항**  
- **transformer_weights.pth** 파일은 Project 2의 학습된 모델 가중치를 포함하고 있습니다.   

---