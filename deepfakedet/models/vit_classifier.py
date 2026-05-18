from typing import Any

import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from torch import nn
from transformers import ViTModel


class ViTClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        peft_cfg: LoraConfig | None = None,
        num_classes: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.backbone_name = backbone_name
        self.peft_cfg = peft_cfg
        self.num_classes = num_classes

        self.model: ViTModel | PeftModel | PeftMixedModel = ViTModel.from_pretrained(
            backbone_name,
        )
        hidden_size = self.model.config.hidden_size

        if self.peft_cfg is not None:
            self.model = get_peft_model(model=self.model, peft_config=self.peft_cfg)

        self.head = nn.Linear(hidden_size, self.num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)  # type: ignore[no-any-return]
