from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch


def weight_init_normal(module, model):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)



class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanPoolingLayer(nn.Module):
    def __init__(self, 
        hidden_size,
        target_size,
        dropout = 0,
    ):
        super(MeanPoolingLayer, self).__init__()
        self.pool = MeanPooling()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, target_size),
            nn.Sigmoid()
        )
        
    def forward(self, inputs, mask):
        last_hidden_states = inputs[0]
        feature = self.pool(last_hidden_states, mask)
        outputs = self.fc(feature)
        return outputs



class HSLanguageModel(nn.Module):
    def __init__(self,
        backbone = 'microsoft/deberta-v3-small',
        target_size = 1,
        head_dropout = 0,
        reinit_nlayers = 0,
        freeze_nlayers = 0,
        reinit_head = True,
        grad_checkpointing = False,
    ):
        super(HSLanguageModel, self).__init__()
        
        self.config = AutoConfig.from_pretrained(backbone, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(backbone, config=self.config)
        self.head = MeanPoolingLayer(
            self.config.hidden_size,
            target_size,
            head_dropout
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone);
        
        
        if grad_checkpointing == True:
            print('Gradient ckpt enabled')
            self.model.gradient_checkpointing_enable()
            
        if reinit_nlayers > 0:
            # Reinit last n encoder layers
            # [TODO] Check if it is autoencoding model: Bert, Roberta, DistilBert, Albert, XLMRoberta, BertModel
            for layer in self.model.encoder.layer[-reinit_nlayers:]: 
                self._init_weights(layer)
        
        if freeze_nlayers > 0:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:freeze_nlayers].requires_grad_(False)
        
        if reinit_head:
            # Reinit layers in head
            self._init_weights(self.head)
        
        
    def _init_weights(self, layer):
        for module in layer.modules():
            init_fn = weight_init_normal
            init_fn(module, self)
    

    def forward(self, inputs):
        outputs = self.model(**inputs)
        outputs = self.head(outputs, inputs['attention_mask'])
        return outputs


if __name__ == '__main__':
    
    model = HSLanguageModel()
    
    
    
    
