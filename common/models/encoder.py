import torch.nn as nn
from transformers import BertModel

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"


class BertEncoder(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, freeze: bool = True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]       #[CLS]トークンの埋め込みを返す
        #文章を[B(文の数),L(トークンの数),768(ベクトルの次元)]に変換する