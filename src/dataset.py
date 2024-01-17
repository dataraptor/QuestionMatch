from torch.utils.data import Dataset
import torch
from utils import read_yaml
import random
import itertools
from tqdm import tqdm

class BanglaHSData(Dataset):
    def __init__(self, data, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.data = []
        for x in data:
            self.data.append({'question': x['question']+x['question_aug'], 'answer': x['answer']})
        
        self.pairs = []
        for idx, d in enumerate(tqdm(self.data)):
            comb = list(itertools.combinations(d['question'], 2))
            
            
            neg_list = self.data[:idx] + self.data[idx+1:]
            for idx, i in enumerate(comb):
                neg = random.sample(random.sample(neg_list, k=1)[0]['question'], k=1)[0]
                comb[idx] += (neg,)
                
            self.pairs.extend(comb)
            
        
    def __len__(self):
        return len(self.pairs)
    
    
    def tokz(self, text):
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, padding='max_length',
            truncation=True,
            return_offsets_mapping=False
        )
        for k, v in inputs.items(): inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs
    
    
    def __getitem__(self, idx):
        #print(self.pairs[idx])
        anchor, pos, neg = self.pairs[idx]
        
        #print(anchor, pos, neg)
        return self.tokz(anchor), self.tokz(pos), self.tokz(neg)
    


# class BanglaHSData(Dataset):
#     def __init__(self, data, tokenizer, max_len):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.max_len = max_len
        
#         self.data = []
#         for x in data:
#             self.data.append({'question': x['question']+x['question_aug'], 'answer': x['answer']})
        
        
        
#     def __len__(self):
#         return len(self.data)
    
    
#     def tokz(self, text):
#         inputs = self.tokenizer(
#             text, 
#             max_length=self.max_len, padding='max_length',
#             truncation=True,
#             return_offsets_mapping=False
#         )
#         for k, v in inputs.items(): inputs[k] = torch.tensor(v, dtype=torch.long)
#         return inputs
    
    
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         anchor = random.sample(sample['question'], k=1)[0]
#         pos = random.sample(sample['question'], k=1)[0]
        
#         neg_list = self.data[:idx] + self.data[idx+1:]
#         neg = random.sample(random.sample(neg_list, k=1)[0]['question'], k=1)[0]
#         # print(anchor, pos, neg)
#         return self.tokz(anchor), self.tokz(pos), self.tokz(neg)
    




if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    backbone = "sagorsarker/bangla-bert-base"
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    
    data = read_yaml('./Data/corpus_aug.yml')['pairs']
    ds = BanglaHSData(data, tokenizer, 512)
    _ = ds[0]
    
