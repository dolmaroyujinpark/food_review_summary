import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AdamW, AutoTokenizer, get_cosine_schedule_with_warmup
from dataset import ReviewDataset
from preprocess import load_data, preprocess_data
from tqdm import tqdm

def train():
    # 설정
    epochs = 3
    batch_size = 1  # 배치 크기 줄임
    lr = 1e-6  # 학습률 조정

    # 기존 학습된 KoT5 모델 불러오기
    model_name = "../models/trained_model_1216part1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 데이터 로드 및 전처리
    raw_data = load_data("../data/1217_data.json")
    processed_data = preprocess_data(
        raw_data,
        tokenizer,
        max_input_length=512,
        max_target_length=128
    )
    dataset = ReviewDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 옵티마이저와 스케줄러
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 학습 루프
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 출력 라벨에서 패딩 토큰 처리
            labels[labels == tokenizer.pad_token_id] = -100

            # 손실 계산
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # NaN 손실 처리
            if torch.isnan(loss):
                print("Skipping step due to NaN loss")
                optimizer.zero_grad()
                continue

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 프로그레스 바에 손실 값 업데이트
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    # 모델 저장 경로 변경
    save_path = "../models/new_trained_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel training complete and saved at {save_path}!")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # GPU 캐시 정리
    train()
