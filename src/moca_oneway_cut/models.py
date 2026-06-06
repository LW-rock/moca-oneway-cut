import torch
import torch.nn as nn


class ScalarFeatureTokenizer(nn.Module):
    def __init__(self, p, d_model):
        super().__init__()
        self.p = p
        self.embeds = nn.ModuleList([nn.Linear(1, d_model) for _ in range(p)])
        self.pos = nn.Parameter(torch.randn(1, p, d_model) * 0.02)

    def forward(self, X):
        tokens = []
        for j in range(self.p):
            xj = X[:, j:j + 1]
            tokens.append(self.embeds[j](xj).unsqueeze(1))
        return torch.cat(tokens, dim=1) + self.pos


class TreatmentQueryModel(nn.Module):
    def __init__(
        self,
        p,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        gate_temp=1.0,
    ):
        super().__init__()
        self.gate_temp = gate_temp
        self.linear_tokenizer = ScalarFeatureTokenizer(p, d_model)
        self.nonlinear_tokenizer = ScalarFeatureTokenizer(p, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.self_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.treat_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_linear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_nonlinear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, 32), nn.GELU(), nn.Linear(32, 2))
        self.ps_head = nn.Sequential(nn.Linear(2 * d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def _make_query(self, X):
        batch_size = X.size(0)
        return self.treat_query.expand(batch_size, -1, -1)

    def forward(self, X):
        h_linear = self.linear_tokenizer(X)
        h_nonlinear = self.self_encoder(self.nonlinear_tokenizer(X))
        q = self._make_query(X)
        z_linear, attn_linear = self.attn_linear(q, h_linear, h_linear)
        z_nonlinear, attn_nonlinear = self.attn_nonlinear(q, h_nonlinear, h_nonlinear)
        z_linear, z_nonlinear = z_linear.squeeze(1), z_nonlinear.squeeze(1)
        gate_logits = self.gate(torch.cat([z_linear, z_nonlinear], dim=-1))
        gate = torch.softmax(gate_logits / self.gate_temp, dim=-1)
        w_linear, w_nonlinear = gate[:, 0:1], gate[:, 1:2]
        fusion = torch.cat([w_linear * z_linear, w_nonlinear * z_nonlinear], dim=-1)
        logit = self.ps_head(fusion).squeeze(-1)
        ps = torch.sigmoid(logit)
        z_treat = w_linear * z_linear + w_nonlinear * z_nonlinear
        return {
            "ps": ps,
            "logit": logit,
            "z_treat": z_treat,
            "z_treat_token": z_treat.unsqueeze(1),
            "gate": gate,
            "H_linear": h_linear,
            "H_nonlinear": h_nonlinear,
            "attn_linear": attn_linear,
            "attn_nonlinear": attn_nonlinear,
        }


class OutcomeFusionModel(nn.Module):
    def __init__(
        self,
        p,
        treatment_model,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        gate_temp=1.0,
    ):
        super().__init__()
        self.treatment_model = treatment_model
        self.gate_temp = gate_temp
        self.linear_tokenizer = ScalarFeatureTokenizer(p, d_model)
        self.nonlinear_tokenizer = ScalarFeatureTokenizer(p, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.self_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.query0 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.query1 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn0_linear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn0_nonlinear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn0_treat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn1_linear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn1_nonlinear = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn1_treat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate0 = nn.Sequential(nn.Linear(3 * d_model, 32), nn.GELU(), nn.Linear(32, 3))
        self.gate1 = nn.Sequential(nn.Linear(3 * d_model, 32), nn.GELU(), nn.Linear(32, 3))
        self.head0 = nn.Sequential(nn.Linear(3 * d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head1 = nn.Sequential(nn.Linear(3 * d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def _one_arm_forward(self, q, h_linear, h_nonlinear, h_treat, arm=0):
        batch_size = h_linear.size(0)
        q = q.expand(batch_size, -1, -1)
        if arm == 0:
            z_linear, attn_linear = self.attn0_linear(q, h_linear, h_linear)
            z_nonlinear, attn_nonlinear = self.attn0_nonlinear(q, h_nonlinear, h_nonlinear)
            z_treat, attn_treat = self.attn0_treat(q, h_treat, h_treat)
            gate_layer, head = self.gate0, self.head0
        else:
            z_linear, attn_linear = self.attn1_linear(q, h_linear, h_linear)
            z_nonlinear, attn_nonlinear = self.attn1_nonlinear(q, h_nonlinear, h_nonlinear)
            z_treat, attn_treat = self.attn1_treat(q, h_treat, h_treat)
            gate_layer, head = self.gate1, self.head1
        z_linear, z_nonlinear, z_treat = z_linear.squeeze(1), z_nonlinear.squeeze(1), z_treat.squeeze(1)
        raw_cat = torch.cat([z_linear, z_nonlinear, z_treat], dim=-1)
        gate = torch.softmax(gate_layer(raw_cat) / self.gate_temp, dim=-1)
        w_linear, w_nonlinear, w_treat = gate[:, 0:1], gate[:, 1:2], gate[:, 2:3]
        fusion = torch.cat([w_linear * z_linear, w_nonlinear * z_nonlinear, w_treat * z_treat], dim=-1)
        pred = head(fusion).squeeze(-1)
        return pred, gate, fusion, attn_linear, attn_nonlinear, attn_treat

    def _treatment_forward(self, X):
        with torch.no_grad():
            return self.treatment_model(X)

    def forward(self, X):
        h_linear = self.linear_tokenizer(X)
        h_nonlinear = self.self_encoder(self.nonlinear_tokenizer(X))
        treat_out = self._treatment_forward(X)
        h_treat = treat_out["z_treat_token"]
        ps = treat_out["ps"]
        treat_gate = treat_out["gate"]
        mu0, gate0, fusion0, a0_linear, a0_nonlinear, a0_treat = self._one_arm_forward(
            self.query0, h_linear, h_nonlinear, h_treat, arm=0
        )
        mu1, gate1, fusion1, a1_linear, a1_nonlinear, a1_treat = self._one_arm_forward(
            self.query1, h_linear, h_nonlinear, h_treat, arm=1
        )
        return {
            "mu0": mu0,
            "mu1": mu1,
            "tau": mu1 - mu0,
            "ps": ps,
            "gate0": gate0,
            "gate1": gate1,
            "treat_gate": treat_gate,
            "fusion0": fusion0,
            "fusion1": fusion1,
            "attn0_linear": a0_linear,
            "attn0_nonlinear": a0_nonlinear,
            "attn0_treat": a0_treat,
            "attn1_linear": a1_linear,
            "attn1_nonlinear": a1_nonlinear,
            "attn1_treat": a1_treat,
        }
