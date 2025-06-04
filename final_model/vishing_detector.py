import torch
from transformers import AutoTokenizer, BertConfig
from final_model.vishing_model import VishingIndoBERTClassifier
from final_model.text_utils import clean_text, tail_tokenizer

class VishingDetector:
    def __init__(self, model_path='final_model/model/', tokenizer_name='indobenchmark/indobert-base-p1', threshold=0.5, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        config = BertConfig.from_pretrained(model_path)
        self.model = VishingIndoBERTClassifier.from_pretrained(model_path, config=config)
        self.model.eval()
        self.threshold = threshold
        self.max_length = max_length

    def predict(self, text):
        clean = clean_text(text)
        tokenized = tail_tokenizer(clean, self.tokenizer, max_length=self.max_length)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            prob = torch.sigmoid(outputs.logits).squeeze().item()
            pred = int(prob > self.threshold)

        label = 'Vishing' if pred == 1 else 'Non-vishing'
        return {
            'text': text,
            'cleaned': clean,
            'predicted_label': label,
            'probability': prob
        }

    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]