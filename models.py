import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, global_mean_pool

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBranch(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, pad_idx, dropout):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    def forward(self, x):
        mask = (x == self.pad_idx)
        x = self.pos(self.emb(x) * math.sqrt(self.d_model))
        x = self.transformer(x, src_key_padding_mask=mask)
        return x

class CrossLayerAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x

class MAHFusion(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_modalities, num_attn_layers, num_heads, dropout):
        super().__init__()
        self.attn_layers = nn.ModuleList([CrossLayerAttention(hidden_dim, num_heads, dropout) for _ in range(num_attn_layers)])
        self.layer_weights = nn.Parameter(torch.ones(num_layers * num_modalities, 1, 1))
        self.weight_norm = nn.Softmax(dim=0)
    def forward(self, fused_all):
        x = fused_all
        for attn in self.attn_layers:
            x = attn(x)
        weights = self.weight_norm(self.layer_weights)
        fused = (x * weights).sum(dim=0)
        return fused, weights

class MultiModalMAHEncoder(nn.Module):
    def __init__(self, graph_dim, fp_dim, mordred_dim, sm_vocab_size, sm_pad_idx,
                 hidden_dim, num_layers, dropout,
                 gnn_hidden, gnn_heads, gnn_dropout,
                 trans_d_model, trans_nhead, trans_num_layers, trans_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.sm_pad_idx = sm_pad_idx
        self.graph_conv_first = GATConv(graph_dim, gnn_hidden, heads=gnn_heads, dropout=gnn_dropout, edge_dim=6)
        self.graph_conv = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.graph_conv.append(GATConv(gnn_hidden * gnn_heads, gnn_hidden, heads=gnn_heads, dropout=gnn_dropout, edge_dim=6))
        self.graph_pool = global_mean_pool
        self.graph_proj = nn.Linear(gnn_hidden * gnn_heads, hidden_dim)
        self.graph_dropout = nn.Dropout(dropout)
        self.seq_first = TransformerBranch(sm_vocab_size, d_model=trans_d_model, nhead=trans_nhead,
                                           num_layers=trans_num_layers, pad_idx=sm_pad_idx, dropout=trans_dropout)
        self.seq_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.seq_layers.append(nn.TransformerEncoderLayer(d_model=trans_d_model, nhead=trans_nhead,
                                                              dim_feedforward=trans_d_model*4, dropout=trans_dropout,
                                                              batch_first=True))
        self.seq_proj = nn.Linear(trans_d_model, hidden_dim)
        self.seq_dropout = nn.Dropout(dropout)
        self.mordred_first = nn.Sequential(nn.Linear(mordred_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.mordred_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_layers - 1)])
        self.mordred_dropout = nn.Dropout(dropout)
        self.fp_first = nn.Sequential(nn.Linear(fp_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.fp_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_layers - 1)])
        self.fp_dropout = nn.Dropout(dropout)
        self.num_modalities = 4

    def forward(self, batch):
        graphs, seqs, mordreds, fingerprints = batch
        seq_mask = (seqs == self.sm_pad_idx)
        layer_outputs = []
        h_graph, h_seq, h_mordred, h_fp = None, None, None, None
        feats = []
        x = F.elu(self.graph_conv_first(graphs.x, graphs.edge_index, edge_attr=graphs.edge_attr))
        x = self.graph_dropout(x)
        g_feat = self.graph_proj(self.graph_pool(x, graphs.batch))
        feats.append(g_feat)
        h_graph = x
        h_seq = self.seq_first(seqs)
        s_feat = self.seq_proj(h_seq.mean(dim=1))
        s_feat = self.seq_dropout(s_feat)
        feats.append(s_feat)
        m_feat = self.mordred_first(mordreds)
        m_feat = self.mordred_dropout(m_feat)
        feats.append(m_feat)
        h_mordred = m_feat
        f_feat = self.fp_first(fingerprints)
        f_feat = self.fp_dropout(f_feat)
        feats.append(f_feat)
        h_fp = f_feat
        layer_outputs.append(torch.stack(feats, dim=0))

        for li in range(self.num_layers - 1):
            feats = []
            x = F.elu(self.graph_conv[li](h_graph, graphs.edge_index, edge_attr=graphs.edge_attr))
            x = self.graph_dropout(x)
            g_feat = self.graph_proj(self.graph_pool(x, graphs.batch))
            feats.append(g_feat)
            h_graph = x
            h_seq = self.seq_layers[li](h_seq, src_key_padding_mask=seq_mask)
            s_feat = self.seq_proj(h_seq.mean(dim=1))
            s_feat = self.seq_dropout(s_feat)
            feats.append(s_feat)
            m_feat = self.mordred_layers[li](h_mordred)
            m_feat = self.mordred_dropout(m_feat)
            feats.append(m_feat)
            h_mordred = m_feat
            f_feat = self.fp_layers[li](h_fp)
            f_feat = self.fp_dropout(f_feat)
            feats.append(f_feat)
            h_fp = f_feat
            layer_outputs.append(torch.stack(feats, dim=0))

        fused_all = torch.cat(layer_outputs, dim=0)
        return fused_all

class MultiModalMAHClassifier(nn.Module):
    def __init__(self, graph_dim, fp_dim, mordred_dim, sm_vocab_size, sm_pad_idx,
                 hidden_dim, num_layers, dropout,
                 gnn_hidden, gnn_heads, gnn_dropout,
                 trans_d_model, trans_nhead, trans_num_layers, trans_dropout,
                 fusion_attn_layers, fusion_num_heads, fusion_dropout
                 ):
        super().__init__()
        self.encoder = MultiModalMAHEncoder(
            graph_dim, fp_dim, mordred_dim, sm_vocab_size, sm_pad_idx,
            hidden_dim, num_layers, dropout,
            gnn_hidden, gnn_heads, gnn_dropout,
            trans_d_model, trans_nhead, trans_num_layers, trans_dropout
        )
        self.fusion = MAHFusion(hidden_dim, num_layers, self.encoder.num_modalities,
                                num_attn_layers=fusion_attn_layers, num_heads=fusion_num_heads, dropout=fusion_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, batch):
        fused_all = self.encoder(batch)
        fused, _ = self.fusion(fused_all)
        logits = self.classifier(fused).squeeze(1)
        return logits