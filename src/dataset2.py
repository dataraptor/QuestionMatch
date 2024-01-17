from torch.utils.data import Dataset
import torch

class BanglaHSData(Dataset):
    def __init__(self, df, tokenizer, max_len, cat2ind):
        super().__init__()
        self.cat2ind = cat2ind
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.data = list(self.df['text'].values)
        self.label = list(self.df['label'].values)
        self.label = [self.cat2ind[x] for x in self.label]
        
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, padding='max_length',
            truncation=True,
            return_offsets_mapping=False
        )
        for k, v in inputs.items(): inputs[k] = torch.tensor(v, dtype=torch.long)
        
        target = self.label[idx]
        target = torch.tensor(target).long()
        return inputs, target

