import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmehr.models.common.linear_finetuner import LinearFinetuner


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.1))
            self.attention_b.append(nn.Dropout(0.1))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 4, 50, 64
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class MultimodalFusion(LinearFinetuner):
    def __init__(self, in_ts_size, in_cxr_size, shared_emb_dim=128, 
                 num_classes=2, n_proto=100, lr=1e-3, *args, **kwargs):
        super().__init__(in_size=in_ts_size)
        self.save_hyperparameters()

        self.proj_ts = nn.Linear(in_ts_size, shared_emb_dim)
        self.proj_cxr = nn.Linear(in_cxr_size, shared_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=shared_emb_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # These layers can be replaced into MLP
        # Attention pooling ...
        self.atten_net = Attn_Net_Gated(L=shared_emb_dim, D=64, dropout=True, n_classes=1)
        self.classifier = nn.Linear(shared_emb_dim, num_classes)

    def forward(self, ts_embs, cxr_embs):
        ts_embs = self.proj_ts(ts_embs)
        cxr_embs = self.proj_cxr(cxr_embs)
        embs = torch.cat([ts_embs, cxr_embs], dim=1)
        embs = self.transformer_encoder(embs)

        # A, _ = self.atten_net(embs)
        # pool_embs = (A.permute(0, 2, 1) @ embs).squeeze(dim=1)
        pool_embs = torch.mean(embs, dim=1)
        logits = self.classifier(pool_embs)

        return logits

    def training_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)    
        step_output = {
            "logits": logits,
            "y": y
        }
        self.val_step_outputs.append(step_output)

    def test_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)
        step_output = {
            "logits": logits,
            "y": y
        }
        self.test_step_outputs.append(step_output)


if __name__ == "__main__":
    ts_emb = torch.rand(4, 16, 257)
    cxr_emb = torch.rand(4, 16, 257)
    model = MultimodalFusion(in_ts_size=257, in_cxr_size=257)
    logits = model(ts_emb, cxr_emb)
    ipdb.set_trace()