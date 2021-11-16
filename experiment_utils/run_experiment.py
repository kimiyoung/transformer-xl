import random
import time
import torch, os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class data_loader:
    def __init__(self, task_name, path='data', batch_size=32, none_mask=True):
        self.X, self.y = np.load(f'{path}/{task_name}_X.npy'), np.load(f'{path}/{task_name}_y.npy')
        self.data_size = self.X.shape[0]
        self.data_ptr = 0

        if none_mask:
            self.src_mask, self.tgt_mask = None, None
        else:
            self.src_masks, self.tgt_mask = np.load(f'{path}/{task_name}_mask.npy'), None

        self.batch_size = batch_size
        self.none_mask = none_mask

    def __next__(self):
        X = self.X[self.data_ptr: self.data_ptr+self.batch_size]
        y = self.y[self.data_ptr: self.data_ptr+self.batch_size]
        
        if not self.none_mask:
            sm = self.src_masks[self.data_ptr: self.data_ptr+self.batch_size]
            sm = torch.tensor(sm).cuda()
        else:
            sm = None
            
        self.data_ptr = (self.data_ptr + self.batch_size) % self.data_size

        return torch.tensor(X).cuda(), torch.tensor(y).cuda(), sm, self.tgt_mask

def save_checkpoint(save_path, model, optim, i, config):
    model.train()
    state = {
        'batch_num': i,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'config': config
    }
    torch.save(state, save_path)            

WINDOW_SIZE = 4
PATIENCE = 10
def train_validate_model(model, train_generator, val_generator, optim, model_name, config, generate_every=1e3, num_batches=1e4, verbose=True, overfit_stop=True, print_file=None, tag='', head_start=15):
    
    fix_seeds()
    t0 = time.time()
    
    log_dir = 'logs/' + model_name.split('_')[0]
    writer = SummaryWriter(log_dir=log_dir)
    if print_file is None:
        print_file = f"{log_dir}/{model_name}_cout_log.txt"

    validation_scores = []
    for i in range(num_batches):

        model.train()
        
        src, tgt, src_mask, tgt_mask = next(train_generator)
        loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss.backward()

        loss_value = loss.item()        
        writer.add_scalars("/train/loss", {model_name: loss_value}, i)
#         if loss_value < 1e-10:
#             break

        optim.step()
        optim.zero_grad()

        if i != 0 and i % generate_every == 0:
            model.eval()
            
            with torch.no_grad():
                src, tgt, src_mask, tgt_mask = next(val_generator)
                
                val_loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                val_loss_value = val_loss.item()
                
                tgt = tgt[:, 1:]
                start_tokens = tgt[:1, :1]

                if src_mask is not None:
                    sm = src_mask[:1]
                else: 
                    sm = src_mask

                predictions = []
                for s, t in zip(src, tgt):
                    sample = model.generate(s[None], start_tokens, config['dec_max_seq_len'], src_mask=sm)
                    predictions.append(sample)
                predictions = torch.cat(predictions)

            num_correct = ((predictions == tgt) & (tgt != 0)).float().sum()
            accuracy = num_correct / (tgt != 0).float().sum()
            writer.add_scalars("/val/accuracy", {model_name: accuracy}, i)
            writer.add_scalars("/val/loss", {model_name: val_loss_value}, i)

            validation_scores.append(accuracy.cpu())            
    
            if verbose:
                with open(print_file, 'a') as f:
                    f.write(f"\n\ninput:  {s}")
                    f.write(f"\npredicted output:  {sample}")
                    f.write(f"\ncorrect output:  {t}")
                    f.write(f"\naccuracy: {accuracy}")
                    f.write(f"\ntime: {round(time.time() - t0)}")
                    t0 = time.time()
            
            # save checkpoint
            if max(validation_scores) == validation_scores[-1]:
                os.system(f'mkdir checkpoints/{model_name.split("_")[0]}')
                os.system(f'mkdir checkpoints/{model_name.split("_")[0]}/validation')
                save_path = f'checkpoints/{model_name.split("_")[0]}/validation/{model_name}_{tag}_maxval.pt'
                save_checkpoint(save_path, model, optim, i, config)
                
            if i // generate_every < head_start:
                continue
                
            # early stopping
            smoothed_val_scores = [np.mean(validation_scores[i-WINDOW_SIZE+1:i]) for i in range(WINDOW_SIZE-1, len(validation_scores))]
            
            if overfit_stop and max(smoothed_val_scores) > max(smoothed_val_scores[-PATIENCE:]):
                break
                
    # save checkpoint
    save_path = f'checkpoints/{model_name.split("_")[0]}/{model_name}_{tag}.pt'
    os.system(f'mkdir checkpoints/{model_name.split("_")[0]}')
    save_checkpoint(save_path, model, optim, i, config)

    writer.flush()


def test_model(model, test_generator, model_name, param, task_name, tag, log_path='logs/_test_results.csv'):
    fix_seeds()
    model.eval()

    src, tgt, src_mask, _ = next(test_generator)
    tgt = tgt[:, 1:]
    start_tokens = tgt[:1, :1]
    if src_mask is not None:
        sm = src_mask[:1]
    else: 
        sm = src_mask

    num_correct = 0
    total_batch_len = 0
    for s, t in zip(src, tgt):
        sample = model.generate(s[None], start_tokens, param['dec_max_seq_len'], src_mask=sm)
        num_correct += torch.abs(((t == sample) & (t != 0)).float()).sum()
        total_batch_len += (t != 0).float().sum()

    accuracy = num_correct / total_batch_len

    param['tag'] = tag
    param['task_name'] = task_name
    param['model_name'] = model_name
    param['accuracy'] = accuracy.cpu().item()

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = df.append(param, ignore_index=True)
    else: 
        df = pd.DataFrame([param])
    df.to_csv(log_path, index=False)
    
    
def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.deterministic = True
    torch.cuda.benchmark = False