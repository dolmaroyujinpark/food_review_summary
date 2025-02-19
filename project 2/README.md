project/
    main.py            # 프로젝트의 메인 실행 파일. 전체 워크플로우의 진입점 역할을 한다.
    requirements.txt   # 프로젝트에서 필요한 Python 패키지 목록. `pip install -r requirements.txt`로 설치 가능.
    generate.py        # 데이터 또는 모델 기반의 결과물을 생성하는 스크립트.
    prepare_spm.py     # 데이터 준비 및 사전학습 모델(SentencePiece) 관련 작업을 수행하는 스크립트.
    
    src/
        dataset.py      # 데이터셋 로드 및 전처리 관련 기능을 정의한 파일.
        evaluate.py     # 모델 평가와 성능 분석 관련 코드가 포함된 파일.
        tokenizer.py    # 텍스트 데이터를 처리하고 토큰화하는 기능을 구현한 파일.
        train.py        # 모델 학습을 위한 주요 로직과 학습 루프가 구현된 파일.
        transformer.py  # Transformer 모델 아키텍처를 정의한 파일.
    
    data/
        README.md       # 데이터 디렉터리에 대한 설명을 제공하는 문서.
        corpus.txt      # 학습 또는 테스트 데이터로 사용될 텍스트 파일.
        combined_corpus.txt # 여러 텍스트 데이터를 통합한 파일.
        spm.model       # SentencePiece 모델 파일.
        spm.vocab       # SentencePiece에서 생성된 어휘 파일.