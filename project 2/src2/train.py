import random
import torch
from tqdm import tqdm

def train_model(model, train_loader, optimizer, criterion, epochs, device, tokenizer):
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        total_loss = 0

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

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        # 에폭이 끝난 후, 무작위로 하나의 배치를 선택하여 결과 출력
        random_batch = random.choice(train_loader.dataset)  # train_loader에서 무작위 샘플 선택
        src_sample = random_batch["src1"].unsqueeze(0).to(device)  # 배치를 추가하여 모델 입력과 맞춤
        trg_sample = random_batch["trg"].to(device)

        model.eval()
        with torch.no_grad():
            output_sample = model(src_sample, trg_sample[:, :-1])
            output_sample = output_sample.argmax(dim=-1)

        # 토큰을 텍스트로 변환
        input_text = tokenizer.decode(src_sample.squeeze().cpu().numpy().tolist())
        target_text = tokenizer.decode(trg_sample.squeeze().cpu().numpy().tolist())
        output_text = tokenizer.decode(output_sample.squeeze().cpu().numpy().tolist())

        print(f"\n[Epoch {epoch}]")
        print(f"Input: {input_text}")
        print(f"Target: {target_text}")
        print(f"Output: {output_text}\n")

        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")
