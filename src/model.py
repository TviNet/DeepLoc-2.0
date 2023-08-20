from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attr_prior import *
from src.constants import *

pos_weights_bce = torch.tensor([1,1,1,3,2.3,4,9.5,4.5,6.6,7.7,32])
def focal_loss(input, target, gamma=1):
    bceloss = F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weights_bce.to(input.device), reduction="none")
    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)
    # compute the loss
    focal_loss = ( (1-pt) ** gamma ) * bceloss
    return focal_loss.mean()

class AttentionHead(nn.Module):
      def __init__(self, hidden_dim, n_heads):
          super(AttentionHead, self).__init__()
          self.n_heads = n_heads
          self.hidden_dim = hidden_dim
          self.preattn_ln = nn.LayerNorm(hidden_dim//n_heads)
          self.Q = nn.Linear(hidden_dim//n_heads, n_heads, bias=False)
          torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1/(hidden_dim//n_heads))

      def forward(self, x, np_mask, lengths):
          # input (batch, seq_len, embed)
          n_heads = self.n_heads
          hidden_dim = self.hidden_dim
          x = x.view(x.size(0), x.size(1), n_heads, hidden_dim//n_heads)
          x = self.preattn_ln(x)
          mul = (x * \
                self.Q.weight.view(1, 1, n_heads, hidden_dim//n_heads)).sum(-1) \
                #* np.sqrt(5)
                #/ np.sqrt(hidden_dim//n_heads)
          mul_score_list = []
          for i in range(mul.size(0)):
              # (1, L) -> (1, 1, L) -> (1, L) -> (1, L, 1)
              mul_score_list.append(F.pad(smooth_tensor_1d(mul[i, :lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),(0, mul.size(1)-lengths[i]),"constant").squeeze(0))
          
          mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
          mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))
          
          attns = F.softmax(mul, dim=1) # (b, l, nh)
          x = (x * attns.unsqueeze(-1)).sum(1)
          x = x.view(x.size(0), -1)
          return x, attns.squeeze(2)

class BaseModel(pl.LightningModule):
    def __init__(self, embed_dim) -> None:
        super().__init__()
       
        self.initial_ln = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)
        self.lr = 1e-3

    def forward(self, embedding, lens, non_mask):
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_pool, x_attns
    
    def attn_reg_loss(self, y_true, y_attn, y_tags, lengths, n):
        loss = 0
        count = 0
        reg_loss = 0
        for i in range(y_true.size(0)):
            reg_loss += fourier_att_prior_loss_dct(
                  F.pad(y_attn[i, :lengths[i]].unsqueeze(0).unsqueeze(0), (8,8),"replicate").squeeze(1),
                  lengths[i]//6,
                  0.2, 3)
        reg_loss = reg_loss / y_true.size(0)
        kld_loss = 0
        kld_count = 0
        for i in range(y_true.size(0)):
            if y_tags[i].sum() > 0:         
                for j in range(9):
                    if (j+1) in y_tags[i]:
                        pos_tar = (y_tags[i]==(j+1)).float()
                        kld_count += 1
                        kld_loss += pos_weights_annot[j] * self.kld(
                            torch.log(y_attn[i, :+lengths[i]].unsqueeze(0)), 
                            pos_tar[:lengths[i]].unsqueeze(0) / pos_tar[:lengths[i]].sum().unsqueeze(0))
        return reg_loss, kld_loss / torch.tensor(kld_count + 1e-5), kld_count

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": [p for n, p in self.named_parameters()]}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=1,
                    min_lr=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "bce_loss"
        }

    def training_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('train_loss_batch', loss)
        return {'loss': loss}

    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, y_tags, _ = batch
        y_pred, y_attns =  self.forward(x, l, n)
        reg_loss, seq_loss, seq_count = self.attn_reg_loss(y, y_attns, y_tags, l, n)
        bce_loss = focal_loss(y_pred, y)
        loss = bce_loss + SUP_LOSS_MULT * seq_loss + REG_LOSS_MULT * reg_loss
        self.log('val_loss_batch', loss)
        return {'loss': loss, 
                'seq_loss': seq_loss,
                'reg_loss': reg_loss,
                'bce_loss': bce_loss,
                'seq_count': seq_count}
  
    def on_val_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

        avg_loss = torch.stack([x['bce_loss'] for x in outputs]).mean()
        self.log('bce_loss', avg_loss, prog_bar=True)
    

class ProtT5Frozen(BaseModel):
    def __init__(self):
        super().__init__(1024)

class ESM1bFrozen(BaseModel):
    def __init__(self):
        super().__init__(1280)
        

pos_weights_annot = torch.tensor([0.23, 0.92, 0.98, 2.63, 5.64, 1.60, 2.37, 1.87, 2.03])
class SignalTypeMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(267, 32)
        self.ln2 = nn.Linear(32, 9)
        self.lr = 1e-3

    def forward(self, x):
        x = nn.Tanh()(self.ln1(x))
        x = self.ln2(x)
        return x

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": [p for n, p in self.named_parameters()], 'lr': self.lr},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        #self.unfreeze()
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights_annot.to(y_pred.device))(y_pred, y)
        self.log('train_loss_batch', loss)
        return {'loss': loss}

    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        #self.freeze()
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights_annot.to(y_pred.device))(y_pred, y)
        self.log('val_loss_batch', loss)
        return {'loss': loss}
  
    def on_val_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
