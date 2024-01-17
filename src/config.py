import os


class CFG:
    group = 'DeBERTa'    # Exp name
    name = 'DEBUG'      # Sub exp name
    amp = True
    
    DIR = "./Data/bengali_hate_v2.0"
    TRAIN_VAL_DF = f"{DIR}/train.csv"
    
    test_size = 0.15
    valid_size = 0.15
    test_split_seed = 42
    valid_split_seed = 42
    
    n_epochs = 6
    batch_size = 2
    max_len = 16
    
    encoder_lr = 2.0e-5
    decoder_lr = 2.0e-5
    
    
    backbone = 'microsoft/deberta-v3-small'
    reinit_nlayers = 0
    freeze_nlayers = 0
    reinit_head = True
    head_dropout = 0
    grad_checkpointing = False
    dropout = 0.2
    weight_decay = 0.01
    
    awp = False
    awp_lr = 1.0e-4
    awp_eps = 1.0e-2
    awp_epoch_start = 1 
    
    fgm = False
    fgm_epoch_start = 1
    
    num_workers = len(os.sched_getaffinity(0))

