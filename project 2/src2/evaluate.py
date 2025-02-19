from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import torch
import nltk
import re

nltk.download('punkt')

def clean_text(text):

    # 1. 반복되는 기호 제거
    text = re.sub(r'([.!?])\1+', r'\1', text).strip()
    # 2. 문장 패턴에 맞지 않는 내용 제거 (한글과 기호 포함된 문장만 유지)
    sentences = re.findall(r'[가-힣a-zA-Z0-9,.\'"\s]+[.!?]', text)
    # 3. 정리된 문장 합치기
    return " ".join(sentences)


def evaluate_model_with_sentencepiece(loader, model, tokenizer, device, num_samples=5):
    model.eval()
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    sample_results = []  # 테스트 샘플 결과 저장

    with torch.no_grad():
        for batch in loader:
            src = batch["src1"].to(device)
            trg = batch["trg"].to(device)

            output = model(src, trg[:, :-1])
            output = torch.argmax(output, dim=2)

            for pred, target in zip(output, trg):
                pred_text = tokenizer.decode(pred.tolist())
                target_text = tokenizer.decode(target.tolist())

                # 불필요한 문장 및 기호 정리
                pred_text = clean_text(pred_text)
                target_text = clean_text(target_text)

                # ROUGE 계산
                rouge_score = rouge_scorer_instance.score(pred_text, target_text)
                for key in rouge_scores.keys():
                    rouge_scores[key].append(rouge_score[key].fmeasure)

                # BLEU 계산
                pred_tokens = pred_text.split()
                target_tokens = target_text.split()
                bleu_scores.append(sentence_bleu([target_tokens], pred_tokens))

                # 샘플 결과 저장
                if len(sample_results) < num_samples:  # 지정된 수만큼 샘플 저장
                    sample_results.append({
                        "Prediction": pred_text,
                        "Reference": target_text
                    })

    # 샘플 결과 출력
    print("\nSample Predictions:")
    for i, result in enumerate(sample_results):
        print(f"\nSample {i + 1}:")
        print(f"Prediction: {result['Prediction']}")
        print(f"Reference: {result['Reference']}")

    # 시각화
    visualize_evaluation_results(rouge_scores, bleu_scores)
    return rouge_scores, bleu_scores


def visualize_evaluation_results(rouge_scores, bleu_scores):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    plt.figure(figsize=(12, 6))

    for metric in metrics:
        plt.plot(rouge_scores[metric], label=f"{metric} (Avg: {sum(rouge_scores[metric]) / len(rouge_scores[metric]):.4f})")
    plt.plot(bleu_scores, label=f"BLEU (Avg: {sum(bleu_scores) / len(bleu_scores):.4f})")

    plt.title("Evaluation Metrics Over Batches")
    plt.xlabel("Batch Index")
    plt.ylabel("Score")
    plt.legend()
    plt.show()