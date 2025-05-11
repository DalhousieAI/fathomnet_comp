# HML Classifier code and get_constr_out 
# from Constrained Feed-Forward Neural Network for HML
# (Coherent Hierarchical Multi-Label Classification Networks - GPL-3.0 License)
# https://github.com/EGiunchiglia/C-HMCNN

import torch
import torch.nn as nn

def get_constr_out(x, R):
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R"""
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out

class FathomNetModel(nn.Module):
    def __init__(
            self,
            encoder,
            classifier,
    ):
        super(FathomNetModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
class OneHotClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(OneHotClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, non_lin="relu"):
        super(ConstrainedFFNNModel, self).__init__()
        fc = []

        if len(hidden_dim) == 0:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, output_dim))
        else:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(len(hidden_dim)):
                if i == len(hidden_dim) - 1:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], output_dim))
                else:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        self.fc = nn.ModuleList(fc)
        self.sigmoid = torch.nn.Sigmoid()

        self.drop = nn.Dropout(dropout)

        if non_lin == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        module_len = len(self.fc)
        for i in range(module_len):
            if i == module_len - 1:
                x = self.fc[i](x)
                x = self.sigmoid(x)
            else:
                if i % 2 == 0:
                    x = self.f(self.fc[i](x))
                else:
                    x = self.f(self.fc[i](x))
                    x = self.drop(x)
        return x

class FusedClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

# ───────────────────────────────────────────────────────────────
#   Attention‑fusion module  (ROI query  ←  context key/value)
# ───────────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, roi_vec: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        """
        roi_vec, ctx_vec : (B, D)
        returns fused     : (B, D)
        """
        # Treat ROI as a single‑token *query*, context as single‑token key/value
        q = roi_vec.unsqueeze(1)  # (B, 1, D)
        k = ctx_vec.unsqueeze(1)
        v = ctx_vec.unsqueeze(1)

        attn_out, _ = self.attn(q, k, v)      # (B, 1, D)
        fused = self.ln(attn_out.squeeze(1) + roi_vec)  # residual
        return fused