import torch
from torch.utils.data import DataLoader, Dataset


n_labels = 4888

class Dataset(Dataset):
    def __init__(self, path_documents, path_labels, tokenizer, max_len, version):
        self.version = version # "train" (features & labels) / "test" (features & labels) / "valid" (features)
        self.n_labels = 4888 if version != "valid" else 1223
        ## TODO valid

        # Labels
        self.labels = [] # Class labels
        self.valid_id = []
        is_kept = []
        with open(path_labels, "r") as f1:
            for line in f1:
                s1, s2 = line.strip().split(',')
                if len(s2.strip())>0:
                    self.labels.append(int(s2))
                    is_kept.append(version != "valid")
                else :
                    self.valid_id.append(s1)
                    is_kept.append(version == "valid")


        # Protein sequences
        self.max_len = max_len # Maximum sequence length threshold, max in train database is 989
        self.tokenizer = tokenizer # Language model tokenizer
        
        documents = [] 
        with open(path_documents, "r") as f1:
            for line in f1:
                documents.append(' '.join(list(line[:-1])))
                
        self.documents = [documents[i] for i in range(len(documents)) if is_kept[i]] # String protein sequences

        if version=='valid':
            self.labels = [-1]*self.n_labels #dummy labels for valid

        print(self.n_labels, len(self.labels),len(self.documents))

        assert len(self.labels) == len(self.documents)

        # Graphe features
        # Inplace with getitem's indexs

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        # print(index)

        # Load protein sequence
        sequence = self.documents[index].split()
        if len(sequence) > self.max_len - 1:
            sequence = sequence[:self.max_len-1]
            
        # Tokenize the sequence
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

        # Label
        target = self.labels[index]

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'indexs': index,
            "target": torch.tensor(target),
        }
        return sample

def get_loader(path_documents='data/sequences.txt', path_labels='data/graph_labels.txt', 
                tokenizer=None, max_len=600, batch_size=64, shuffle=False, version=None, drop_last=False):
        
    dataset = Dataset(path_documents=path_documents, path_labels=path_labels, tokenizer=tokenizer, max_len=600, version=version)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=drop_last)
    return data_loader