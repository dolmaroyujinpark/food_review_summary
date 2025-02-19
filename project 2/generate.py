import torch
from src2.dataset2 import load_json
from src2.tokenizer import SentencePieceTokenizer
from src2.transformer import SparseTransformer


def top_k_top_p_sampling(logits, top_k=50, top_p=0.9):
    """Top-K 및 Top-p 샘플링"""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()


def enhanced_decode(model, src, tokenizer, device, max_length=50, top_k=50, top_p=0.9):
    """향상된 디코딩 (Top-K 및 Top-p 샘플링 사용)"""
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    enc_src = model.encoder(src, src_mask)

    start_token = tokenizer.sp.piece_to_id("<s>")
    end_token = tokenizer.sp.piece_to_id("</s>")
    trg = torch.tensor([[start_token]], dtype=torch.long).to(device)

    summary_tokens = []

    for _ in range(max_length):
        trg_mask = model.make_trg_mask(trg)
        output = model.decoder(trg, enc_src, src_mask, trg_mask)
        logits = output[:, -1, :].squeeze(0)

        next_token = top_k_top_p_sampling(logits, top_k, top_p)

        if next_token == end_token:
            break

        summary_tokens.append(next_token)
        trg = torch.cat([trg, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

    return tokenizer.decode(summary_tokens)


def initialize_model_and_tokenizer(model_path, spm_model_path, device):
    tokenizer = SentencePieceTokenizer(spm_model_path)
    model = SparseTransformer(
        src_vocab_size=tokenizer.sp.get_piece_size(),
        trg_vocab_size=tokenizer.sp.get_piece_size(),
        src_pad_idx=tokenizer.sp.piece_to_id("<pad>"),
        trg_pad_idx=tokenizer.sp.piece_to_id("<pad>"),
        device=device,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer


def process_first_review(test_json_path, model, tokenizer, device, max_length=50, top_k=50, top_p=0.9):
    """첫 번째 리뷰에 대해 요약 생성"""
    test_reviews, _ = load_json(test_json_path)
    if not test_reviews:
        print("테스트 데이터가 비어 있습니다.")
        return

    # 첫 번째 리뷰만 처리
    review = test_reviews[3]
    print("\n[Original Review]")
    print(review)

    src_tokens = torch.tensor([tokenizer.encode(review)], dtype=torch.long)
    print("\n[Enhanced Decoding (Top-K & Top-p)]")
    summary = enhanced_decode(model, src_tokens, tokenizer, device, max_length, top_k, top_p)
    print(summary)


if __name__ == "__main__":
    # 경로 설정
    MODEL_PATH = "./transformer_weights.pth"
    SPM_MODEL_PATH = "./data/spm.model"
    TEST_JSON_PATH = "./data/test_final.json"

    # 디바이스 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 토크나이저 초기화
    model, tokenizer = initialize_model_and_tokenizer(MODEL_PATH, SPM_MODEL_PATH, device)

    # 첫 번째 리뷰 처리 및 출력
    process_first_review(TEST_JSON_PATH, model, tokenizer, device, max_length=50, top_k=50, top_p=0.9)
