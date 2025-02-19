import torch
from torch.utils.data import Dataset

def load_json(filepath):
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 리뷰 리스트를 병합하여 저장
    reviews = [" ".join(item["reviews"]) for item in data if "reviews" in item and "summary" in item]
    summaries = [item["summary"] for item in data if "reviews" in item and "summary" in item]

    # 데이터 쌍의 길이 확인
    assert len(reviews) == len(summaries), "Reviews and summaries lengths do not match!"
    return reviews, summaries


class SentencePieceDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_len=100):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, index):
        src = self.tokenizer.encode(self.texts[index], self.max_len)
        trg = self.tokenizer.encode(self.summaries[index], self.max_len)
        return {
            "src1": torch.tensor(src, dtype=torch.long),
            "trg": torch.tensor(trg, dtype=torch.long),
        }


