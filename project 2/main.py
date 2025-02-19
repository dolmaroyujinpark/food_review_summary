from sklearn.model_selection import train_test_split
from src2.dataset2 import load_json, SentencePieceDataset
from src2.tokenizer import SentencePieceTokenizer
from src2.transformer import SparseTransformer
from src2.evaluate import evaluate_model_with_sentencepiece
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm

# SentencePiece 모델 로드
spm_model_path = "./data/spm.model"
tokenizer = SentencePieceTokenizer(spm_model_path)

# 데이터 로드
json_path = "./data/1215_data.json"
reviews, summaries = load_json(json_path)

# 데이터 분리 (train:80%, test:20%)
train_reviews, test_reviews, train_summaries, test_summaries = train_test_split(
    reviews, summaries, test_size=0.2, random_state=42
)

# 데이터셋 및 DataLoader 구성
train_dataset = SentencePieceDataset(train_reviews, train_summaries, tokenizer)
test_dataset = SentencePieceDataset(test_reviews, test_summaries, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Transformer 모델 생성
if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA 사용
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # MPS 사용 (Apple Silicon)
else:
    device = torch.device("cpu")  # CPU fallback

print(f"Using device: {device}")

model = SparseTransformer(
    src_vocab_size=tokenizer.sp.get_piece_size(),
    trg_vocab_size=tokenizer.sp.get_piece_size(),
    src_pad_idx=tokenizer.sp.piece_to_id("<pad>"),
    trg_pad_idx=tokenizer.sp.piece_to_id("<pad>"),
    device=device,
)
model.to(device)

# 학습 설정
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.sp.piece_to_id("<pad>"))
epochs = 10

# 가중치 저장 관련 초기값
best_loss = float("inf")  # 최적 손실 초기값

# 학습 과정
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    model.train()
    train_loss = 0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        src = batch["src1"].to(device)
        trg = batch["trg"].to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.reshape(-1, output.shape[2])
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    average_train_loss = train_loss / len(train_loader)
    print(f"Training Loss: {average_train_loss:.4f}")

    # 최적 모델 저장 (최저 손실 기준)
    if average_train_loss < best_loss:
        best_loss = average_train_loss
        torch.save(model.state_dict(), "transformer_weights.pth")
        print("Best model saved!")

# 학습 종료 후 테스트 데이터로 최종 평가
print("\nEvaluating on Test Set...")
# 저장된 가중치 불러오기
model.load_state_dict(torch.load("transformer_weights.pth"))
print("Loaded the best model for evaluation.")

rouge_scores_test, bleu_scores_test = evaluate_model_with_sentencepiece(
    test_loader, model, tokenizer, device, num_samples=5
)

# 평가 결과 요약
print("\nTest Evaluation Results:")
print(f"Average ROUGE-1 Score: {sum(rouge_scores_test['rouge1']) / len(rouge_scores_test['rouge1']):.4f}")
print(f"Average ROUGE-2 Score: {sum(rouge_scores_test['rouge2']) / len(rouge_scores_test['rouge2']):.4f}")
print(f"Average ROUGE-L Score: {sum(rouge_scores_test['rougeL']) / len(rouge_scores_test['rougeL']):.4f}")
print(f"Average BLEU Score: {sum(bleu_scores_test) / len(bleu_scores_test):.4f}")
