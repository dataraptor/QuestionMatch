from config import CFG
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import HSLanguageModel
from dataset import BanglaHSData
from utils import AverageMeter
import pandas as pd
import os, argparse
import torch, wandb
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

def make_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--group', type=str, default=CFG.group, help='Group of the experiment; (default: %(default)s)')
    arg('--name', type=str, default=CFG.name, help='Name of the experiment; (default: %(default)s)')
    # hyper parameters
    arg('--n_epochs', type=int, default=CFG.n_epochs, help='Number of Epoch; (default: %(default)s)')
    arg('--batch_size', type=int, default=CFG.batch_size, help='Batch Size; (default: %(default)s)')
    arg('--encoder_lr', type=float, default=CFG.encoder_lr, help='Encoder Learning Rate; (default: %(default)s)')
    arg('--decoder_lr', type=float, default=CFG.decoder_lr, help='Decoder Learning Rate; (default: %(default)s)')
    arg('--max_len', type=int, default=CFG.max_len, help='Max Input length; (default: %(default)s)')
    # Model
    arg('--backbone', type=str, default=CFG.backbone, help='Backbone; (default: %(default)s)')
    arg('--head_dropout', type=float, default=CFG.head_dropout, help='Head Dropout; (default: %(default)s)')
    arg('--reinit_nlayers', type=int, default=CFG.reinit_nlayers, help='Reinit Last N Layer; (default: %(default)s)')
    arg('--freeze_nlayers', type=int, default=CFG.freeze_nlayers, help='Freeze First N Layer; (default: %(default)s)')
    arg('--reinit_head', type=bool, default=CFG.reinit_head, help='Reinit Head; (default: %(default)s)')
    arg('--grad_checkpointing', type=bool, default=CFG.grad_checkpointing, help='Grad Checkpointing; (default: %(default)s)')
    arg('--weight_decay', type=float, default=CFG.weight_decay, help='Weight Decay; (default: %(default)s)')
    #optimizer
    arg('--awp', type=bool, default=CFG.awp, help='Enable AWP; (default: %(default)s)')
    arg('--awp_lr', type=float, default=CFG.awp_lr, help='AWP Learning Rate; (default: %(default)s)')
    arg('--awp_eps', type=float, default=CFG.awp_eps, help='AWP EPS; (default: %(default)s)')
    arg('--awp_epoch_start', type=int, default=CFG.awp_epoch_start, help='AWP Start Epoch; (default: %(default)s)')
    arg('--fgm', type=bool, default=CFG.fgm, help='Enable FGM; (default: %(default)s)')
    arg('--fgm_epoch_start', type=int, default=CFG.fgm_epoch_start, help='FGM Start Epoch; (default: %(default)s)')
    # others
    arg('--num_workers', type=int, default=CFG.num_workers, help='Number of dataloader workers; (default: %(default)s)')
    arg("--debug", action="store_true", help="debug")
    arg("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Which device to train on; (default: %(default)s)")
    return parser



class AWP:
    def __init__(
        self,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        apex: bool,
        adv_param: str="weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01,
        epoch_start: int=1
    ) -> None:
        isinstance(apex, bool)
        isinstance(adv_lr, float)
        isinstance(adv_eps, float)
        isinstance(epoch_start, int)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}
        self.epoch_start = epoch_start

    def attack_backward(self, inputs: dict, label: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step()
            y_preds = self.model(inputs)
            adv_loss = self.criterion(y_preds, label)       # Modified for cross entropy loss
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
    

class FGM():
    def __init__(self, model, epoch_start):
        isinstance(epoch_start, int)
        self.model = model
        self.backup = {}
        self.epoch_start = epoch_start

    def attack(self, epsilon = 1., emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


def train(epoch, data, model, optimizer, loss_fn, scaler, awp, fgm):
    inputs, targets = data
    
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=CFG.amp):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()
    
    ## FGM
    if fgm != None and fgm.epoch_start <= epoch+1:
        #print('fgm attach')
        fgm.attack()
        with torch.cuda.amp.autocast(enabled=CFG.amp):
            y_preds = model(inputs)
            loss = loss_fn(y_preds, targets)
            scaler.scale(loss).backward()
        fgm.restore()
    
    ## AWP
    if awp != None and awp.epoch_start <= epoch+1:
        loss = awp.attack_backward(inputs, targets)
        scaler.scale(loss).backward()
        awp._restore()
    
    scaler.step(optimizer)
    scaler.update()
    
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc


@torch.no_grad()
def validate(data, model, loss_fn):
    model.eval()
    inputs, targets = data
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc, outputs.to('cpu').numpy(), targets.to('cpu').numpy()



if __name__ == '__main__':
    args = make_parser().parse_args()
    device = args.device
    #torch.multiprocessing.set_start_method('spawn')
    print('num_workers', args.num_workers)
    
    wandb.login()
    run = wandb.init(
        project='HateGuard',
        name=args.name,
        config = {
            'n_epochs': args.n_epochs,
            'encoder_lr': args.encoder_lr,
            'decoder_lr': args.decoder_lr,
            'batch_size': args.batch_size, 
            'max_len': args.max_len,
            'model': {
                'backbone': args.backbone,
                'head_dropout': args.head_dropout,
                'reinit_nlayers': args.reinit_nlayers,
                'freeze_nlayers': args.freeze_nlayers,
                'reinit_head': args.reinit_head,
                'grad_checkpointing': args.grad_checkpointing,
                'weight_decay': args.weight_decay,
            },
            'awp': {
                'awp': args.awp,
                'awp_lr': args.awp_lr,
                'awp_eps': args.awp_eps,
                'awp_epoch_start': args.awp_epoch_start,
            },
            'fgm': {
                'fgm': args.fgm,
                'fgm_epoch_start': args.fgm_epoch_start,
            },
            'dataset': {
                'test_size': CFG.test_size,
                'valid_size': CFG.valid_size,
                'test_split_seed': CFG.test_split_seed,
                'valid_split_seed': CFG.valid_split_seed,
            }
        },
        group=args.group,
        anonymous=None
    )
    
    
    df=pd.read_csv(CFG.TRAIN_VAL_DF)
    ind2cat = sorted(df['label'].unique().tolist())
    cat2ind = {cat: ind for ind, cat in enumerate(ind2cat)}
    
    train_val_data, test_data = train_test_split(
        df, 
        test_size=CFG.test_size, 
        shuffle=True, 
        random_state=CFG.test_split_seed
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=CFG.valid_size, 
        shuffle=True, 
        random_state=CFG.valid_split_seed
    )
    
    print('Train Size', len(train_data))
    print('Val Size', len(val_data))
    
    # Debug
    if args.debug:
        train_data = train_data[:16]
        val_data = val_data[:8]
        test_data = test_data[:8]
    
    
    model = HSLanguageModel(
        backbone = args.backbone,
        target_size = len(ind2cat),
        head_dropout = args.head_dropout,
        reinit_nlayers = args.reinit_nlayers,
        freeze_nlayers = args.freeze_nlayers,
        reinit_head = args.reinit_head,
        grad_checkpointing = args.grad_checkpointing,
    ).to(device)
    
    
    
    train_ds = BanglaHSData(train_data, model.tokenizer, args.max_len, cat2ind)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False
    )
    
    val_ds = BanglaHSData(val_data, model.tokenizer, args.max_len, cat2ind)
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False
    )
    
    test_ds = BanglaHSData(test_data, model.tokenizer, args.max_len, cat2ind)
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
          'lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
          'lr': args.encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
          'lr': args.decoder_lr, 'weight_decay': 0.0}
    ]
    
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(optimizer_parameters, eps=1.0e-6, betas=(0.9, 0.999))
    num_train_steps = int(len(train_ds) / args.batch_size * args.n_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps, num_cycles=0.5
    )
    
    
    awp = AWP(
        model = model,
        criterion=loss_fn,
        optimizer=optimizer,
        apex=CFG.amp,
        adv_lr=args.awp_lr,
        adv_eps=args.awp_eps,
        epoch_start=args.awp_epoch_start,
    )
    if args.awp == False: awp = None
    
    fgm = FGM(
        model = model,
        epoch_start = args.fgm_epoch_start
    )
    if args.fgm == False: fgm = None
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.amp)
    for epoch in tqdm(range(args.n_epochs), desc='EPOCH'):
        
        # Train Model
        train_losses, train_accs = AverageMeter(), AverageMeter()
        n_batch = len(train_dl)
        for i, (inputs, targets) in enumerate(tqdm(train_dl, desc="TRAIN")):
            for k, v in inputs.items(): inputs[k] = inputs[k].to(device)
            targets = targets.to(device)
            
            train_loss, train_acc = train(epoch, (inputs, targets), model, optimizer, loss_fn, scaler, awp, fgm)
            scheduler.step()
            
            train_losses.update(train_loss.item(), n_batch)
            train_accs.update(train_acc, n_batch)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, 'train_step':epoch*len(train_dl)+i})
        
        # Validate Model
        val_losses, valid_accs = AverageMeter(), AverageMeter()
        n_batch = len(val_dl)
        for i, (inputs, targets) in enumerate(tqdm(val_dl, desc="VALID")):
            for k, v in inputs.items(): inputs[k] = inputs[k].to(device)
            targets = targets.to(device)
            
            val_loss, val_acc, pred, target = validate((inputs, targets), model, loss_fn)
            val_losses.update(val_loss.item(), n_batch)
            valid_accs.update(val_acc, n_batch)
            wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_acc": val_acc, 'val_step':epoch*len(val_dl)+i})
        
        # Log Results
        print(f'\nEPOCH: {epoch+1: >2}  train_loss:{train_losses.avg: .3f}  valid_loss:{val_losses.avg: .3f}  train_acc:{train_accs.avg: .4f}  valid_accs:{valid_accs.avg: .4f}\n')
        wandb.log({
            "epoch": epoch+1,
            "train_loss(avg)": train_losses.avg,
            "valid_loss(avg)": val_losses.avg,
            "train_acc(avg)": train_accs.avg,
            "valid_acc(avg)": valid_accs.avg,
            "learning_rate": scheduler.get_last_lr(),
        })
        
    
    # Calculate Test Accuracy
    test_accs = AverageMeter()
    n_batch = len(test_dl)
    for i, (inputs, targets) in enumerate(test_dl):
        for k, v in inputs.items(): inputs[k] = inputs[k].to(device)
        targets = targets.to(device)
        
        _, test_acc, pred, target = validate((inputs, targets), model, loss_fn)
        test_accs.update(test_acc, n_batch)
    wandb.log({"test_acc": test_accs.avg})
    
    
    if not os.path.exists('./Output'): os.mkdir('./Output')
    torch.save(model.state_dict(), './Output/model_weights.pth')
    pd.DataFrame({'labels': ind2cat}).to_csv('./Output/model_ind2cat.csv')
    
    wandb.finish()

