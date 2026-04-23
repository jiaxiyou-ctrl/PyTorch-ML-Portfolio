"""Language encoder that uses CLIP's text backbone to produce
fixed-dimensional embeddings for task instructions.
"""

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Dict


class LanguageEncoder(nn.Module):

    def __init__(
        self,                                                        
        output_dim: int = 256,                                       
        clip_model_name: str = "openai/clip-vit-base-patch32",      
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()                                          

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.text_model.parameters():
                param.requires_grad = False

        clip_dim = self.text_model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._cache: Dict[str, torch.Tensor] = {}                  

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.projection.parameters()).device           
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            clip_output = self.text_model(**tokens)                  
            text_features = clip_output.pooler_output

        projected = self.projection(text_features)
        return projected

    def get_embedding(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            embedding = self.encode_text([text])
            self._cache[text] = embedding.squeeze(0).detach()

        return self._cache[text]

    def precompute_all(self, task_texts: List[str]) -> Dict[str, torch.Tensor]:
        embeddings = self.encode_text(task_texts)

        result = {}
        for i, text in enumerate(task_texts):
            self._cache[text] = embeddings[i].detach()
            result[text] = embeddings[i].detach()

        return result
