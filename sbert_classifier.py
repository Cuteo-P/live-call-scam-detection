import torch.nn as nn
from sentence_transformers import SentenceTransformer

class SBERTClassifier(nn.Module):
    def __init__(self, sbert_model_name='sbert_model_v3', num_classes=2):
        super().__init__()
        self.sbert = SentenceTransformer(sbert_model_name)
        self.classifier = nn.Linear(self.sbert.get_sentence_embedding_dimension(), num_classes)

    def forward(self, texts):
        device = self.classifier.weight.device
        features = self.sbert.tokenize(texts)
        for key in features:
            features[key] = features[key].to(device)
        embeddings = self.sbert.forward(features)['sentence_embedding']
        return self.classifier(embeddings)