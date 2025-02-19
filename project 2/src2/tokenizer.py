import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, max_len=100):
        tokens = self.sp.encode_as_ids(text)
        tokens = tokens[:max_len - 2]
        tokens = [self.sp.piece_to_id("<s>")] + tokens + [self.sp.piece_to_id("</s>")]
        return tokens + [self.sp.piece_to_id("<pad>")] * (max_len - len(tokens))

    def decode(self, tokens):
        return self.sp.decode_ids(tokens)