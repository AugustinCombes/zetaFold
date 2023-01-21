import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes, device='cuda', PRE_TRAINED_MODEL_NAME = 'yarongef/DistilProtBert'):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes).to(device)
        
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = output.last_hidden_state[:, 0, :]
        output = nn.Dropout(0.2)(output) # 0.3 meilleure submission
        output = nn.ReLU()(output)
        output = self.classifier(output)
        # output = nn.ReLU()(output)
        # return output

        return nn.LogSoftmax(dim=1)(output)