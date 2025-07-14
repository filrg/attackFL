import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from collections import OrderedDict
from torch.utils.data import Dataset

class ICUData(Dataset):
    def __init__(self, dataframe, vitals_cols, labs_cols, label_col):
        self.vitals = dataframe[vitals_cols].values  # Lấy dữ liệu Vitals
        self.labs = dataframe[labs_cols].values  # Lấy dữ liệu Labs
        self.labels = dataframe[label_col].values  # Lấy nhãn

        # Chuyển đổi NaN thành giá trị phù hợp (nếu có)
        self.vitals = torch.tensor(self.vitals, dtype=torch.float32)
        self.labs = torch.tensor(self.labs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vitals[idx], self.labs[idx], self.labels[idx]


class CNNModel(nn.Module):
    '''
    CNN Model for ICU dataset
    '''
    def __init__(self):
        super(CNNModel, self).__init__()

        # Vitals input branch
        self.vitals_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.vitals_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.vitals_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.vitals_pool = nn.AdaptiveAvgPool1d(4)  # Adaptive pooling để fix output shape
        #self.vitals_bn = nn.BatchNorm1d(128)
        self.vitals_dropout = nn.Dropout(0.3)

        # Labs input branch
        self.labs_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.labs_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.labs_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.labs_pool = nn.AdaptiveAvgPool1d(4)  # Adaptive pooling
        #self.labs_bn = nn.BatchNorm1d(128)
        self.labs_dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 4, 128)  # 128 channels * 4 pooled features per branch
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, vitals, labs):
        # Ensure vitals & labs have correct shape
        vitals = vitals.unsqueeze(1)
        labs = labs.unsqueeze(1)

        # Vitals branch
        vitals = F.relu(self.vitals_conv1(vitals))
        vitals = F.relu(self.vitals_conv2(vitals))
        vitals = F.relu(self.vitals_conv3(vitals))
        vitals = self.vitals_pool(vitals)  # (batch_size, 128, 4)
        #vitals = self.vitals_bn(vitals)
        vitals = vitals.view(vitals.shape[0], -1)  # Flatten thành (batch_size, 128*4)
        vitals = self.vitals_dropout(vitals)

        # Labs branch
        labs = F.relu(self.labs_conv1(labs))
        labs = F.relu(self.labs_conv2(labs))
        labs = F.relu(self.labs_conv3(labs))
        labs = self.labs_pool(labs)  # (batch_size, 128, 4)
        #labs = self.labs_bn(labs)
        labs = labs.view(labs.shape[0], -1)  # Flatten
        labs = self.labs_dropout(labs)

        # Concatenate branches
        merged = torch.cat([vitals, labs], dim=1)  # (batch_size, 128 * 2 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.sigmoid(self.output(x))

        return output


class RNNModel(nn.Module):
    '''
    RNN Model for ICU dataset
    '''
    def __init__(self, vitals_input_dim=7, labs_input_dim=16, hidden_dim=32, dropout_rate=0.3):
        super(RNNModel, self).__init__()

        self.mask_value = -2.0
        self.hidden_dim = hidden_dim

        # Vital channel
        self.vitals_gru1 = nn.GRU(vitals_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.vitals_gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.vitals_gru3 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.vitals_ln = nn.LayerNorm(hidden_dim * 2)
        self.vitals_dropout = nn.Dropout(dropout_rate)

        # Labs channel
        self.labs_gru1 = nn.GRU(labs_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.labs_gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.labs_gru3 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.labs_ln = nn.LayerNorm(hidden_dim * 2)
        self.labs_dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, vitals, labs):
        # Masking values for vitals
        vitals = torch.where(vitals == self.mask_value, torch.zeros_like(vitals), vitals)

        # Đảm bảo đầu vào có 3 chiều: (batch_size, sequence_length, feature_dim)
        if vitals.dim() == 2:
            vitals = vitals.unsqueeze(1)  # Thêm chiều seq_length = 1

        # Vital channel GRU layers
        vitals, _ = self.vitals_gru1(vitals)
        vitals, _ = self.vitals_gru2(vitals)
        vitals, _ = self.vitals_gru3(vitals)

        # Lấy hidden state cuối cùng
        vitals = vitals[:, -1, :]  # (batch_size, hidden_dim)
        vitals = self.vitals_ln(vitals)
        vitals = self.vitals_dropout(vitals)

        # Masking values for labs
        labs = torch.where(labs == self.mask_value, torch.zeros_like(labs), labs)

        # Đảm bảo đầu vào có 3 chiều: (batch_size, sequence_length, feature_dim)
        if labs.dim() == 2:
            labs = labs.unsqueeze(1)  # Thêm chiều seq_length = 1

        # Labs channel GRU layers
        labs, _ = self.labs_gru1(labs)
        labs, _ = self.labs_gru2(labs)
        labs, _ = self.labs_gru3(labs)

        # Lấy hidden state cuối cùng
        labs = labs[:, -1, :]  # (batch_size, hidden_dim)
        labs = self.labs_ln(labs)
        labs = self.labs_dropout(labs)

        # Concatenation of both branches
        merged = torch.cat([vitals, labs], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.output(x))

        return output


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate,
                                               batch_first=True)
        self.attention_norm = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, input_dim)
        )
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.dropout2(ffn_output))  # Skip connection

        return x


class TransformerModel(nn.Module):
    '''
    Transformer Model for ICU dataset
    '''
    def __init__(self, vitals_input_dim=7, labs_input_dim=16, num_heads=4, ff_dim=6):
        super(TransformerModel, self).__init__()

        # Vitals branch
        self.vitals_dense = nn.Linear(vitals_input_dim, 64)
        self.vitals_transformer = TransformerBlock(64, num_heads, ff_dim)
        self.vitals_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1)  # Kết hợp cả hai để giữ nhiều thông tin hơn
        )
        self.vitals_bn = nn.LayerNorm(64)  # Thay BatchNorm bằng LayerNorm

        # Labs branch
        self.labs_dense = nn.Linear(labs_input_dim, 64)
        self.labs_transformer = TransformerBlock(64, num_heads, ff_dim)
        self.labs_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1)
        )
        self.labs_bn = nn.LayerNorm(64)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Điều chỉnh kích thước
        self.dropout = nn.Dropout(0.3)  # Giảm dropout để tránh mất dữ liệu
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, vitals, labs):
        vitals = F.gelu(self.vitals_dense(vitals))  # Thay ReLU bằng GELU
        vitals = vitals.unsqueeze(1)
        vitals = self.vitals_transformer(vitals)
        vitals = vitals.squeeze(1)

        vitals = self.vitals_bn(vitals)

        labs = F.gelu(self.labs_dense(labs))
        labs = labs.unsqueeze(1)
        labs = self.labs_transformer(labs)
        labs = labs.squeeze(1)
        labs = self.labs_bn(labs)

        merged = torch.cat([vitals, labs], dim=1)

        x = F.gelu(self.fc1(merged))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        output = torch.sigmoid(self.output(x))

        return output


##### HyperNetwork #####

class HyperNetwork(nn.Module):
    def __init__(self, target_model, n_nodes, embedding_dim, hidden_dim=100, spec_norm=False, n_hidden=2):
        super(HyperNetwork, self).__init__()
        self.target_model = target_model
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.hyper_layers = self.create_hyper_layers(hidden_dim, spec_norm)  # Pass hidden_dim instead of embedding_dim

    def create_hyper_layers(self, embedding_dim, spec_norm):
        """
        Sinh các lớp hypernetwork để tạo tham số tương ứng với target model.
        """
        state_dict = self.target_model.state_dict()
        hyper_layers = nn.ModuleDict()

        for name, param in state_dict.items():
            output_dim = param.numel()
            safe_name = name.replace(".", "__")  # Thay "." bằng "_"
            layer = nn.Linear(embedding_dim, output_dim)
            if spec_norm:
                layer = nn.utils.spectral_norm(layer)  # Áp dụng spectral norm nếu cần
            hyper_layers[safe_name] = layer  # Lưu với tên an toàn

        return hyper_layers

    def forward(self, idx):
        emd = self.embeddings(idx)  # Nhúng ID thành vector embedding
        features = self.mlp(emd)  # Qua MLP để tạo đặc trưng

        # Tạo trọng số từ hyper_layers
        weights = OrderedDict()

        for safe_name, layer in self.hyper_layers.items():
            original_name = safe_name.replace("__", ".")  # Chuyển từ tên an toàn về tên gốc
            weight = layer(features)

            # Kiểm tra nếu layer là bias
            if 'bias' in original_name:
                weights[original_name] = weight.view(-1)  # Flatten nếu là bias
            else:
                # Giữ nguyên cấu trúc ban đầu giống target model
                target_weight_shape = self.target_model.state_dict()[original_name].shape
                weights[original_name] = weight.view(target_weight_shape)

        return weights, emd


#hyper  ver2 

class CNNHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, hidden_dim, n_hidden, spec_norm=False):
        super().__init__()
        # Embedding layer: nhận số lượng node và embedding dimension
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        # Xây dựng MLP (feature extractor)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm
            else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            # Chú ý: dùng hidden_dim -> hidden_dim trong cả nhánh spectral norm và không spectral norm
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm
                else nn.Linear(hidden_dim, hidden_dim)
            )
        self.mlp = nn.Sequential(*layers)

        # Sinh trọng số cho nhánh vitals
        self.vitals_conv1_weights = nn.Linear(hidden_dim, 1 * 32 * 3)
        self.vitals_conv1_bias = nn.Linear(hidden_dim, 32)
        self.vitals_conv2_weights = nn.Linear(hidden_dim, 64 * 32 * 3)
        self.vitals_conv2_bias = nn.Linear(hidden_dim, 64)
        self.vitals_conv3_weights = nn.Linear(hidden_dim, 128 * 64 * 3)
        self.vitals_conv3_bias = nn.Linear(hidden_dim, 128)

        # Sinh trọng số cho nhánh labs
        self.labs_conv1_weights = nn.Linear(hidden_dim, 1 * 32 * 3)
        self.labs_conv1_bias = nn.Linear(hidden_dim, 32)
        self.labs_conv2_weights = nn.Linear(hidden_dim, 64 * 32 * 3)
        self.labs_conv2_bias = nn.Linear(hidden_dim, 64)
        self.labs_conv3_weights = nn.Linear(hidden_dim, 128 * 64 * 3)
        self.labs_conv3_bias = nn.Linear(hidden_dim, 128)

        # Sinh trọng số cho các lớp Fully Connected
        # fc1: shape (128, 1024) vì 128*2*4 = 1024
        self.fc1_weights = nn.Linear(hidden_dim, 128 * (128 * 2 * 4))  # = 128 * 1024 = 131072 elements
        self.fc1_bias = nn.Linear(hidden_dim, 128)
        # fc2: shape (64, 128)
        self.fc2_weights = nn.Linear(hidden_dim, 64 * 128)
        self.fc2_bias = nn.Linear(hidden_dim, 64)
        # fc3: shape (32, 64)
        self.fc3_weights = nn.Linear(hidden_dim, 32 * 64)
        self.fc3_bias = nn.Linear(hidden_dim, 32)
        # output: shape (1, 32)
        self.output_weights = nn.Linear(hidden_dim, 1 * 32)
        self.output_bias = nn.Linear(hidden_dim, 1)

        # Áp dụng spectral normalization nếu cần
        if spec_norm:
            self.vitals_conv1_weights = spectral_norm(self.vitals_conv1_weights)
            self.vitals_conv1_bias = spectral_norm(self.vitals_conv1_bias)
            self.vitals_conv2_weights = spectral_norm(self.vitals_conv2_weights)
            self.vitals_conv2_bias = spectral_norm(self.vitals_conv2_bias)
            self.vitals_conv3_weights = spectral_norm(self.vitals_conv3_weights)
            self.vitals_conv3_bias = spectral_norm(self.vitals_conv3_bias)
           
            self.labs_conv1_weights = spectral_norm(self.labs_conv1_weights)
            self.labs_conv1_bias = spectral_norm(self.labs_conv1_bias)
            self.labs_conv2_weights = spectral_norm(self.labs_conv2_weights)
            self.labs_conv2_bias = spectral_norm(self.labs_conv2_bias)
            self.labs_conv3_weights = spectral_norm(self.labs_conv3_weights)
            self.labs_conv3_bias = spectral_norm(self.labs_conv3_bias)
           
            self.fc1_weights = spectral_norm(self.fc1_weights)
            self.fc1_bias = spectral_norm(self.fc1_bias)
            self.fc2_weights = spectral_norm(self.fc2_weights)
            self.fc2_bias = spectral_norm(self.fc2_bias)
            self.fc3_weights = spectral_norm(self.fc3_weights)
            self.fc3_bias = spectral_norm(self.fc3_bias)
            self.output_weights = spectral_norm(self.output_weights)
            self.output_bias = spectral_norm(self.output_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)            # (batch_size, embedding_dim)
        features = self.mlp(emd)                # (batch_size, hidden_dim)

        weights = OrderedDict({
            # Vitals branch
            "vitals_conv1.weight": self.vitals_conv1_weights(features).view(32, 1, 3),
            "vitals_conv1.bias": self.vitals_conv1_bias(features).view(-1),
            "vitals_conv2.weight": self.vitals_conv2_weights(features).view(64, 32, 3),
            "vitals_conv2.bias": self.vitals_conv2_bias(features).view(-1),
            "vitals_conv3.weight": self.vitals_conv3_weights(features).view(128, 64, 3),
            "vitals_conv3.bias": self.vitals_conv3_bias(features).view(-1),
            
            # Labs branch
            "labs_conv1.weight": self.labs_conv1_weights(features).view(32, 1, 3),
            "labs_conv1.bias": self.labs_conv1_bias(features).view(-1),
            "labs_conv2.weight": self.labs_conv2_weights(features).view(64, 32, 3),
            "labs_conv2.bias": self.labs_conv2_bias(features).view(-1),
            "labs_conv3.weight": self.labs_conv3_weights(features).view(128, 64, 3),
            "labs_conv3.bias": self.labs_conv3_bias(features).view(-1),
      
            # Fully Connected layers
            "fc1.weight": self.fc1_weights(features).view(128, 128 * 2 * 4),  # (128, 1024)
            "fc1.bias": self.fc1_bias(features).view(-1),
            "fc2.weight": self.fc2_weights(features).view(64, 128),
            "fc2.bias": self.fc2_bias(features).view(-1),
            "fc3.weight": self.fc3_weights(features).view(32, 64),
            "fc3.bias": self.fc3_bias(features).view(-1),
            "output.weight": self.output_weights(features).view(1, 32),
            "output.bias": self.output_bias(features).view(-1),
            
                    })
        return weights, emd

### ICU HAR

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # (batch, seq_len, d_model)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=6):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, kernel_size=3, padding=1)  # (Btrain_dataset, d_model, 561)
        self.pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)         # (B, d_model, 561)
        x = x.permute(0, 2, 1)   # (B, 561, d_model)
        x = self.pe(x)           # Add positional encoding
        x = x.permute(1, 0, 2)   # Transformer needs (seq_len, batch, d_model)
        x = self.transformer(x) # (seq_len, batch, d_model)
        x = x.mean(dim=0)       # Global average pooling
        return self.classifier(x)
