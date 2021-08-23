import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.models.joint.utils import interpolate
# from ltr.models.layers.normalization import InstanceL2Norm
# from collections import OrderedDict

def mutual_matching(dot_prod, eps=1e-5):
    dot_prod_A_max, _ = torch.max(dot_prod, dim=2, keepdim=True)
    dot_prod_B_max, _ = torch.max(dot_prod, dim=1, keepdim=True)
    dot_prod_A = dot_prod / (dot_prod_A_max + eps)
    dot_prod_B = dot_prod / (dot_prod_B_max + eps)
    dot_prod = dot_prod * (dot_prod_A * dot_prod_B)
    return dot_prod

def softmax_topk(dot_prod, top=50):
    values, indices = torch.topk(dot_prod, k=top, dim=2)
    x_exp = torch.exp(values)
    x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
    x_exp /= x_exp_sum
    dot_prod.zero_().scatter_(2, indices, x_exp) # B * HW * THW
    return dot_prod

class Attention(nn.Module):
    def __init__(self, feature_dim=512, key_dim=128, tau=1 / 30, topk=False):
        super(Attention, self).__init__()
        self.WK = nn.Linear(feature_dim, key_dim)
        self.WV = nn.Linear(feature_dim, feature_dim)
        self.tau = tau
        self.topk = topk

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0)  # Batch, Dim, Len_1

        w_q = self.WK(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2)  # Batch, Len_2, Dim

        # w_v = self.WV(value)
        w_v = value
        w_v = w_v.permute(1, 0, 2)  # Batch, Len_1, Dim

        dot_prod = torch.bmm(w_q, w_k)      # Batch, Len_2, Len_1
        if self.topk:
            affinity = softmax_topk(dot_prod)
        else:
            affinity = F.softmax(dot_prod / self.tau, dim=-1)

        output = torch.bmm(affinity, w_v)   # Batch, Len_2, Dim
        output = output.permute(1, 0, 2)    # Len_2, Batch, Dim

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, attn, feature_dim):
        super().__init__()
        self.self_attn = attn
        self.norm = nn.InstanceNorm2d(feature_dim)

    def instance_norm(self, src, input_shape):
        num_frames, num_sequences, c, h, w = input_shape
        # Normlization
        src = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, c, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)
        return src

    def forward(self, src, pos=None):
        src_shape = src.shape
        num_frames, num_sequences, c, h, w = src_shape
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)

        if pos is not None:
            pos = pos.view(num_frames, num_sequences, 1, -1).permute(0, 3, 1, 2)
            pos = pos.reshape(-1, num_sequences, 1)

        ## self attention
        # src_attn = self.self_attn(query=src, key=src, value=src)
        # src = src + src_attn
        # src = self.instance_norm(src, src_shape)

        ## Menory saving version self attention
        src_attn = torch.zeros_like(src)
        step_len = h * w
        for i in range(num_frames):
            start = i * step_len
            end = (i+1) * step_len
            src_attn[start:end] = self.self_attn(query=src[start:end], key=src, value=src)
        src = src + src_attn
        src = self.instance_norm(src, src_shape)


        out = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, attn, feature_dim):
        super().__init__()
        self.self_attn = attn
        self.cross_attn = Attention(feature_dim=feature_dim, key_dim=128, tau=1/30, topk=False)
        self.norm = nn.InstanceNorm2d(feature_dim)

    def instance_norm(self, src, input_shape):
        num_frames, num_sequences, c, h, w = input_shape
        # Normlization
        src = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, c, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)
        return src

    def forward(self, tgt, memory, pos=None, pos_enc=None):
        """
        tgt:    1,          num_sequences, c, h, w
        memory: num_frames, num_sequences, c, h, w
        pos:    num_frames, num_sequences, h*16, w*16
        pos_enc:num_frames, num_sequences, ce, h, w
        """
        tgt_shape = tgt.shape
        mem_shape = memory.shape
        num_frames, num_sequences, c, h, w = mem_shape

        tgt = tgt.reshape(1, num_sequences, c, -1).permute(0, 3, 1, 2)
        tgt = tgt.reshape(-1, num_sequences, c)

        memory = memory.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, num_sequences, c)

        if pos_enc is not None:
            pos_enc = pos_enc.reshape(num_frames, num_sequences, -1, h * w).permute(0, 3, 1, 2)
            pos_enc = pos_enc.reshape(num_frames * h * w, num_sequences, -1)

        if pos is not None:
            pos = interpolate(pos, (h, w))
            pos = pos.view(num_frames, num_sequences, 1, -1).permute(0, 3, 1, 2)
            pos = pos.reshape(-1, num_sequences, 1)

        # self-attention
        tgt_attn = self.self_attn(query=tgt, key=tgt, value=tgt)
        tgt = tgt + tgt_attn

        tgt = self.instance_norm(tgt, tgt_shape)

        ### Mask Encoding transform
        enc = self.cross_attn(query=tgt, key=memory, value=pos_enc)
        out = enc.reshape(1, h, w, num_sequences, -1).permute(0, 3, 4, 1, 2)

        return out


class Transformer(nn.Module):
    def __init__(self, feature_dim, key_dim, feature_adjustor=None, feature_extractor=None):
        super().__init__()
        self.feature_adjustor = feature_adjustor    # Extracts features input to the transformer
        self.feature_extractor = feature_extractor

        attn = Attention(feature_dim=feature_dim, key_dim=key_dim)
        # self.modulator = Modulator(feature_dim=feature_dim)
        self.encoder = TransformerEncoder(attn=attn, feature_dim=feature_dim)
        self.decoder = TransformerDecoder(attn=attn, feature_dim=feature_dim)

    def encode(self, train_feat):
        '''
        train_feat: num_frames, num_sequences, c, h, w
        '''
        num_frames, num_sequences, c, h, w = train_feat.shape
        if self.feature_adjustor is not None:
            train_feat_adj = self.feature_adjustor(train_feat.view(-1, *train_feat.shape[-3:]))
            train_feat_adj = train_feat_adj.reshape(num_frames, num_sequences, *train_feat_adj.shape[-3:])
        else:
            train_feat_adj = train_feat

        encoded_train_feat = self.encoder(train_feat_adj, pos=None)

        return encoded_train_feat

    def decode(self, test_feat, encoded_train_feat, train_mask=None, train_mask_enc=None):
        '''
        test_feat:          1, num_sequences, c, h, w
        encoded_train_feat: num_frames, num_sequences, c, h, w
        train_mask:         num_frames, num_sequences, h*16, w*16
        '''
        num_frames, num_sequences, c, h, w = test_feat.shape
        if self.feature_adjustor is not None:
            test_feat_adj = self.feature_adjustor(test_feat.view(-1, *test_feat.shape[-3:]))
            test_feat_adj = test_feat_adj.reshape(num_frames, num_sequences, *test_feat_adj.shape[-3:])
        else:
            test_feat_adj = test_feat

        decoded_mask_enc = self.decoder(test_feat_adj, encoded_train_feat, pos=train_mask, pos_enc=train_mask_enc)

        return decoded_mask_enc

    def forward(self, test_feat, train_feat, train_mask=None, train_mask_enc=None):
        '''
        train_feat: num_frames, num_sequences, c, h, w.
        test_feat:  1, num_sequences, c, h, w
        train_mask: num_frames, num_sequences, h*16, w*16.
        '''
        encoded_train_feat = self.encode(train_feat)

        decoded_mask_enc = self.decode(test_feat, encoded_train_feat, train_mask, train_mask_enc)

        return decoded_mask_enc


if __name__ == "__main__":
    transformer_feature_adjustor = nn.Conv2d(1024, 512, 1, 1)
    transformer_feature_extractor = nn.Conv2d(512, 512, 1, 1)

    tfm = Transformer(512, 128, feature_adjustor=None, feature_extractor=None)
    tfm.cuda()
    train_feat = torch.rand(10, 2, 512, 56, 30).cuda()
    train_mask_enc = torch.rand(10, 2, 64, 56, 30).cuda()
    test_feat = torch.rand(1, 2, 512, 56, 30).cuda()
    test_mask = torch.rand(1, 2, 56*16, 30*16).cuda()
    with torch.no_grad():
        tfm.init(test_feat, test_mask)
        decoded_mask_enc = tfm(test_feat, train_feat, train_mask_enc=train_mask_enc)

    import pdb; pdb.set_trace()