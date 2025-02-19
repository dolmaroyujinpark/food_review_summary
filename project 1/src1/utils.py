import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(path):
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
