import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = "cpu"
# device = "mps"

from utils.transformers import Model, ClassificationHead

from tqdm import trange, tqdm

import numpy
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, log_loss

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn


PRE_TRAINED_MODEL_NAME = 'yarongef/DistilProtBert'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)

class Dataset(Dataset):
    def __init__(self, path_documents, path_labels=None, tokenizer=tokenizer, max_len=600):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.documents = []
        self.labels = []

        with open(path_documents, "r") as f1:
            for line in f1:
                self.documents.append(' '.join(list(line[:-1])))

        with open(path_labels, "r") as f1:
            for line in f1:
                self.labels.append(int(line.strip()))
                
        assert len(self.labels) == len(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        sequence = self.documents[index].split()
        if len(sequence) > self.max_len - 1:
            sequence = sequence[: self.max_len - 1]
            
        # source_sequence = list(map(lambda x: token2ind[x], sequence)) 
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        target = [self.labels[index]]

        sample = {
            # "source_sequence": sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            "target": torch.tensor(target),
        }
        return sample

def get_loader(path_documents='data/train_sequences.txt', path_labels='data/train_graph_labels.txt', 
                tokenizer=tokenizer, max_len=600, batch_size=16, shuffle=False):
        
    dataset = Dataset(path_documents=path_documents, path_labels=path_labels, tokenizer=tokenizer, max_len=600)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)#collate_fn=padder, 
    return data_loader, dataset


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
        self.bert.eval()
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes).to(device)
                            # nn.Sequential(
                            #             nn.Dropout(p=0.2),
                            #             ,
                            #             nn.Tanh())
        
    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.classifier(nn.ReLU()(output.pooler_output))
        return nn.LogSoftmax(dim=1)(output)

model = ProteinClassifier(18)

for module in model.bert.encoder.layer:
    for param in module.parameters():
        param.requires_grad = False


def train(path_data_train='data/train_sequences.txt', path_labels_train='data/train_graph_labels.txt', 
            path_data_valid=None,save_interval=-1,log_interval=5,task="classification",batch_size=32,):

    model.train()
    total_loss = 0.0
    data_loader, _ = get_loader(path_data_train, path_labels_train, tokenizer, batch_size=batch_size, shuffle=True)
    
    losses = []
    for idx, data in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        input = data['input_ids'].to(device)
        src_mask = data['attention_mask'].to(device)
        output = model(input, src_mask)
        
        output = output.view(-1, output.shape[-1])
        target = data['target'].reshape(-1).to(device)
        print('target', target, 'prediction', torch.argmax(output, dim=-1))
        output = output.to(device)

        # print('shapes', output.shape, target.shape)
        loss = criterion(output, target).to(device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent exploding gradient 

        optimizer.step()
        total_loss += loss.item() 
        display_loss = criterion(nn.Softmax(dim=1)(output), target)
        tqdm.write(f'Batch {idx}: last loss {display_loss}')
    return losses

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
lr = 1e-2  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#pretraining
log_interval = 500
epochs = 1
for epoch in range(1, epochs + 1):
    train(
        save_interval=-1,
        task='pr',
        batch_size=16,
        log_interval=log_interval,
    )

torch.save({"state": model.base.state_dict(),}, "1finetuningepoch.pt")