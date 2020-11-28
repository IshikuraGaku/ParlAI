# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 
#s

"""
Implements NN code for transformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Transformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Transformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.core.utils import warn_once
from parlai.core.utils import neginf

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    warn_once("Installing APEX can give a significant speed boost.")
    from torch.nn import LayerNorm

LAYER_NORM_EPS = 1e-12  # Epsilon for layer norm.


def _normalize(tensor, norm_layer):
    """Broadcast layer norm."""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _build_universal_multilayer_encoder(
    opt,
    dictionary,
    embedding=None,
    padding_idx=None,
    reduction_type='mean',
    n_positions=1024,
    n_segments=0,
):
    return UniversalTransformerMultiLayerEncoder(
        n_heads=opt['n_heads'],
        n_layers=4,
        n_rec=2,
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=reduction_type,
        n_positions=n_positions,
        n_segments=n_segments,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
        act_l2=False,
    )


def _build_universal_multilayer_decoder(
    opt, dictionary, embedding=None, padding_idx=None, n_positions=1024, n_segments=0
):
    return UniversalTransformerMultiLayerDecoder(
        n_heads=opt['n_heads'],
        n_layers=4,
        n_rec=2,
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        n_segments=n_segments,
        act_l2=False,
    )

def _build_universal_encoder(
    opt,
    dictionary,
    embedding=None,
    padding_idx=None,
    reduction_type='mean',
    n_positions=1024,
    n_segments=0,
):
    return UniversalTransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=reduction_type,
        n_positions=n_positions,
        n_segments=n_segments,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
        act_l2=True,
        light_act=True,
    )


def _build_universal_decoder(
    opt, dictionary, embedding=None, padding_idx=None, n_positions=1024, n_segments=0
):
    return UniversalTransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        n_segments=n_segments,
        act_l2=True,
        light_act=True,
    )

def _build_encoder(
    opt,
    dictionary,
    embedding=None,
    padding_idx=None,
    reduction_type='mean',
    n_positions=1024,
    n_segments=0,
):
    return TransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=4,
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=reduction_type,
        n_positions=n_positions,
        n_segments=n_segments,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
    )


def _build_decoder(
    opt, dictionary, embedding=None, padding_idx=None, n_positions=1024, n_segments=0
):
    return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=4,
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        n_segments=n_segments,
    )

def gelu(tensor):
    """
    Compute gelu function.

    c.f. https://arxiv.org/abs/1606.08415
    """
    return 0.5 * tensor * (1.0 + th.erf(tensor / math.sqrt(2.0)))


def get_n_positions_from_options(opt):
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            n_positions = 1024
    return n_positions

class TransformerGeneratorModel(TorchGeneratorModel):
    """Implements a full generator model, with one encoder and one decoder."""
    #_build_universal_encoder と _build_encoderで切り替えられるはず

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_universal_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            n_positions=n_positions,
            n_segments=n_segments,
        )
        self.decoder = _build_universal_decoder(
            opt, dictionary, self.embeddings, self.pad_idx, n_positions=n_positions
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not th.is_tensor(indices):
            indices = th.LongTensor(indices).to(enc.device)
        enc = th.index_select(enc, 0, indices)
        mask = th.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder the decoder incremental state.

        Not implemented in Transformers, since ``incremental_state`` is always None.
        """
        # no support for incremental decoding at this time
        return None

    def output(self, tensor):
        """Compute output logits."""
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output


class TransformerMemNetModel(nn.Module):
    """Model which takes context, memories, candidates and encodes them."""

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]

        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.share_word_embedding = opt.get('share_word_embeddings', True)
        if not self.share_word_embedding:
            self.cand_embeddings = _create_embeddings(
                dictionary, opt['embedding_size'], self.pad_idx
            )

        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False
            if not self.share_word_embedding:
                self.cand_embeddings.weight.requires_grad = False

        n_positions = get_n_positions_from_options(opt)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.reduction_type = opt.get('reduction_type', 'mean')
        self.n_segments = opt.get('n_segments', 0)

        
        self.context_encoder = _build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=self.reduction_type,
            n_positions=n_positions,
            n_segments=self.n_segments,
        )

        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            if not self.share_word_embedding:
                cand_embeddings = self.cand_embeddings
            else:
                cand_embeddings = self.embeddings
            self.cand_encoder = _build_universal_encoder(
                opt,
                dictionary,
                cand_embeddings,
                self.pad_idx,
                n_positions=n_positions,
                reduction_type=self.reduction_type,
                n_segments=self.n_segments,
            )

        # build memory encoder
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            self.memory_transformer = self.context_encoder

        self.attender = BasicAttention(
            dim=2, attn=opt['memory_attention'], residual=True
        )

    def encode_cand(self, words):
        """Encode the candidates."""
        if words is None:
            return None

        # flatten if there are many candidates
        if words.dim() == 3:
            oldshape = words.shape
            words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None

        encoded = self.cand_encoder(words)

        if oldshape is not None:
            encoded = encoded.reshape(oldshape[0], oldshape[1], -1)

        return encoded

    def encode_context_memory(self, context_w, memories_w):
        """Encode the memories."""
        # [batch, d]
        if context_w is None:
            # it's possible that only candidates were passed into the
            # forward function, return None here for LHS representation
            return None, None

        context_h = self.context_encoder(context_w)

        if memories_w is None:
            return [], context_h

        bsz = memories_w.size(0)
        memories_w = memories_w.view(-1, memories_w.size(-1))
        memories_h = self.memory_transformer(memories_w)
        memories_h = memories_h.view(bsz, -1, memories_h.size(-1))

        context_h = context_h.unsqueeze(1)
        context_h, weights = self.attender(context_h, memories_h)

        return weights, context_h

    def forward(self, xs, mems, cands):
        """Forward pass."""
        weights, context_h = self.encode_context_memory(xs, mems)
        cands_h = self.encode_cand(cands)

        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)

        return context_h, cands_h


def create_position_codes(n_pos, dim, out):
    """Create positional codes and store them in ``out``."""
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out[:, 0::2] = th.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = th.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


class TransformerResponseWrapper(nn.Module):
    """
    Wrap transformer response.

    Pushes input through transformer and MLP.
    """

    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),  # TODO: should this also be gelu?
            nn.Linear(hdim, dim),
        )

    def forward(self, *args):
        """Forward pass."""
        return self.mlp(self.transformer(*args))


class UniversalTransformerMultiLayerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    #この辺はなるべく変えたくない
    def __init__(
        self,
        n_heads,
        n_layers,
        n_rec,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        act=True,
        res_net=False,
        act_l2=False
    ):
        super(UniversalTransformerMultiLayerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_rec = n_rec
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        self.res_net = res_net
        self.act_l2 = act_l2
        self.act_loss = None
        if res_net:
            self.res_norm = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # Not Error check
        # create the timing embeddings
        # this make each layer embedding
        self.timing_embeddings = nn.Embedding(n_rec, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_rec, embedding_size, out=self.timing_embeddings.weight
            )
        else:
            nn.init.normal_(self.timing_embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)
        
        self.act = act
        if(self.act):
            self.act_fn_layers = nn.ModuleList()
            for _ in range(self.n_layers):
                self.act_fn_layers.append(
                    ACT_basic(self.dim)
                )

        # build the model
        self.enc_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.enc_layers.append(
                UniversalTransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                )
            )
        self.output_scaling = output_scaling

        self.num = 0
        self.num_of_layer_list = th.tensor([0,0,0,0,0,0]).cuda()

    def forward(self, input, positions=None, segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param BoolTensor[batch,seqlen] mask:
            The attention mask; 1 means attend, 0 means ignore.
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=th.int64) - 1).clamp_(min=0)
        #未確認tensror[batch,secLen,emb]
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        #tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        if self.n_segments >= 1:
            if segments is None:
                segments = th.zeros_like(input)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        
        if(self.act):
            if self.res_net:
                res_tensor = tensor.clone()
                for i in range(self.n_layers):
                    tensor, (remainders, n_updates) = self.act_fn_layers[i](tensor, input, mask, self.enc_layers[i], self.timing_embeddings, self.position_embeddings, self.n_rec)
                    tmp_tensor = tensor.clone()
                    tensor = tensor + res_tensor
                    tensor = _normalize(tensor, self.res_norm)
                    res_tensor = tmp_tensor.clone()
                    

            else:
                act_loss_tmp = None
                for i in range(self.n_layers):
                    tensor, (remainders, n_updates) = self.act_fn_layers[i](tensor, input, mask, self.enc_layers[i], self.timing_embeddings, self.position_embeddings, self.n_rec)
                    if act_loss_tmp is None:
                        act_loss_tmp = th.mean(remainders + n_updates)
                    else:
                        act_loss_tmp = act_loss_tmp + th.mean(remainders + n_updates)
                self.act_loss = act_loss_tmp / self.n_layers
                #return tensor, (remainders, n_updates)
            """
            n_update = n_updates.reshape(n_updates.shape[0]*n_updates.shape[1])

            self.num += len(n_update)
            for i in range(self.n_layers):
                self.num_of_layer_list[i] += th.sum((n_update == th.tensor([i+1]).float().cuda()).int())
            
            average = 0
            for i in range(self.n_layers):
                average += (i+1) * self.num_of_layer_list[i]
                print(average)
            average /= self.num
            
            variance = 0
            for i in range(self.n_layers):
                variance += ((i+1 - average) * (i+1 - average) * self.num_of_layer_list[i])
            variance /= self.num
            print(self.num_of_layer_list)
            print("enc average")
            print(average)
            print("enc variance")
            print(variance)
            """
        else:
            ##ここでループここにPosとTimEmbedding
            for i in range(self.n_layers):
                #tensorの形がわかんねえ予想(b, s, emb)
                tensor = tensor + self.position_embeddings(positions).expand_as(tensor)#[s,emb]
                tensor = tensor + self.timing_embeddings(th.tensor([i], device=input.device)).expand_as(tensor)#emb
                tensor = self.enc(tensor, mask)

        tensor = tensor * self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type == 'none' or self.reduction_type is None:
            output = tensor
            return output, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )


class UniversalTransformerMultiLayerEncoderLayer(nn.Module):
    """Implements a single Transformer encoder layer."""
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        """Forward pass."""
        #特に変更を加える必要はないと見た
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        return tensor

class UniversalTransformerMultiLayerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        n_rec,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
        act=True,    #add ACT
        res_net=False,
        act_l2=False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_rec = n_rec
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        self.out_dim = embedding_size
        self.res_net = res_net
        self.act_l2 = act_l2
        self.act_loss = None
        if res_net:
            self.res_norm = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # Not Error check
        # create the timing embeddings
        # this make each layer embedding
        self.timing_embeddings = nn.Embedding(n_rec, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_rec, embedding_size, out=self.timing_embeddings.weight
            )
        else:
            nn.init.normal_(self.timing_embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)
        
        self.act = act
        if(self.act):
            self.act_fn_layers = nn.ModuleList()
            for _ in range(self.n_layers):
                self.act_fn_layers.append(
                    ACT_basic(self.dim)
                )

        # build the model
        
        self.dec_layers = nn.ModuleList()
        for _ in range(self.n_layers): 
            self.dec_layers.append(
                UniversalTransformerDecoderLayer(
                    n_layers,
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )
            )

        self.num = 0
        self.num_of_layer_list = th.tensor([0,0,0,0,0,0]).cuda()

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            Ignored. Should always be ``None`` in this version.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = th.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = self.dropout(tensor + self.position_embeddings(positions).expand_as(tensor))

        if (self.act):
            if self.res_net:
                res_tensor = tensor.clone()
                for i in range(self.n_layers):
                    tensor, (remainders, n_updates) = self.act_fn_layers[i](tensor, input, encoder_mask, self.dec_layers[i], self.timing_embeddings, self.position_embeddings, self.n_rec, encoder_output)
                    tmp_tensor = tensor.clone()
                    tensor = tensor + res_tensor
                    tensor = _normalize(tensor, self.res_norm)
                    res_tensor = tmp_tensor.clone()
            else:
                act_loss_tmp = None
                for i in range(self.n_layers):
                    tensor, (remainders, n_updates) = self.act_fn_layers[i](tensor, input, encoder_mask, self.dec_layers[i], self.timing_embeddings, self.position_embeddings, self.n_rec, encoder_output)
                    if act_loss_tmp is None:
                        act_loss_tmp = th.mean(remainders + n_updates)
                    else:
                        act_loss_tmp = act_loss_tmp + th.mean(remainders + n_updates)
                self.act_loss = act_loss_tmp / self.n_layers
            """
            #tensor, (remainders, n_updates)            
            n_update = n_updates.reshape(n_updates.shape[0]*n_updates.shape[1])

            self.num += len(n_update)
            for i in range(self.n_layers):
                self.num_of_layer_list[i] += th.sum((n_update == th.tensor([i+1]).float().cuda()).int())
            
            average = 0
            for i in range(self.n_layers):
                average += (i+1) * self.num_of_layer_list[i]
            average /= self.num
            
            variance = 0
            for i in range(self.n_layers):
                variance += ((i+1 - average) * (i+1 - average) * self.num_of_layer_list[i])
            variance /= self.num
            
            print(self.num_of_layer_list)
            print("dec average")
            print(average)
            print("dec variance")
            print(variance)
            """
            
            return tensor, (remainders, n_updates)
            #return tensor, None

        else:
            for i in range(self.n_layers):
                #tensorの形がわかんねえ予想(b, s, emb)
                tensor = tensor + self.position_embeddings(positions).expand_as(tensor)#[s,emb]
                tensor = tensor + self.timing_embeddings(th.tensor([i], device=input.device)).expand_as(tensor)#emb
                tensor = self.dec(tensor, encoder_output, encoder_mask)

            return tensor, None


class UniversalTransformerMultiLayerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_layers,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask):
        """Forward pass."""
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = th.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class UniversalTransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    #この辺はなるべく変えたくない
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        act=True,
        act_l2=False,
        light_act=False
    ):
        super(UniversalTransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        self.act_l2 = act_l2
        self.light_act = light_act
        self.act_loss = None
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # Not Error check
        # create the timing embeddings
        # this make each layer embedding
        self.timing_embeddings = nn.Embedding(n_layers, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_layers, embedding_size, out=self.timing_embeddings.weight
            )
        else:
            nn.init.normal_(self.timing_embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)
        
        self.act = act
        if(self.act):
            if self.light_act:
                self.act_fn = ACT_Light(self.dim)
            else:
                self.act_fn = ACT_basic(self.dim)

        # build the model
        self.enc = UniversalTransformerEncoderLayer(
                n_heads,
                embedding_size,
                ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
                variant=variant,
                activation=activation,
            )
        self.output_scaling = output_scaling

        self.num = 0
        self.num_of_layer_list = th.tensor([0,0,0,0,0,0]).cuda()

    def forward(self, input, positions=None, segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param BoolTensor[batch,seqlen] mask:
            The attention mask; 1 means attend, 0 means ignore.
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=th.int64) - 1).clamp_(min=0)
        #未確認tensror[batch,secLen,emb]
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        #tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        if self.n_segments >= 1:
            if segments is None:
                segments = th.zeros_like(input)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)

        if(self.act):
            tensor, (remainders, n_updates) = self.act_fn(tensor, input, mask, self.enc, self.timing_embeddings, self.position_embeddings, self.n_layers)
            self.act_loss = th.mean(remainders + n_updates)
            if self.act_l2:
                tmp_act_l2_loss = None
                for param in self.act_fn.parameters():
                    if tmp_act_l2_loss is None:
                        tmp_act_l2_loss = th.norm(param)
                    else:
                        tmp_act_l2_loss =  tmp_act_l2_loss + th.norm(param)
                self.act_loss = self.act_loss + tmp_act_l2_loss
            """
            n_update = n_updates.reshape(n_updates.shape[0]*n_updates.shape[1])

            self.num += len(n_update)
            for i in range(self.n_layers):
                self.num_of_layer_list[i] += th.sum((n_update == th.tensor([i+1]).float().cuda()).int())
            
            average = 0
            for i in range(self.n_layers):
                average += (i+1) * self.num_of_layer_list[i]
                print(average)
            average /= self.num
            
            variance = 0
            for i in range(self.n_layers):
                variance += ((i+1 - average) * (i+1 - average) * self.num_of_layer_list[i])
            variance /= self.num
            print(self.num_of_layer_list)
            print("enc average")
            print(average)
            print("enc variance")
            print(variance)
            """
        else:
            ##ここでループここにPosとTimEmbedding
            for i in range(self.n_layers):
                #tensorの形がわかんねえ予想(b, s, emb)
                tensor = tensor + self.position_embeddings(positions).expand_as(tensor)#[s,emb]
                tensor = tensor + self.timing_embeddings(th.tensor([i], device=input.device)).expand_as(tensor)#emb
                tensor = self.enc(tensor, mask)

        tensor = tensor * self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type == 'none' or self.reduction_type is None:
            output = tensor
            return output, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

class UniversalTransformerEncoderLayer(nn.Module):
    """Implements a single Transformer encoder layer."""

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        """Forward pass."""
        #特に変更を加える必要はないと見た
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        return tensor


class UniversalTransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
        act=True,    #add ACT
        act_l2=False,
        light_act=False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        self.out_dim = embedding_size
        self.act_loss = None
        self.act_l2 = act_l2
        self.light_act = light_act
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # Not Error check
        # create the timing embeddings
        # this make each layer embedding
        self.timing_embeddings = nn.Embedding(n_layers, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_layers, embedding_size, out=self.timing_embeddings.weight
            )
        else:
            nn.init.normal_(self.timing_embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)
        
        self.act = act
        if(self.act):
            if self.light_act:
                self.act_fn = ACT_Light(self.dim)
            else:
                self.act_fn = ACT_basic(self.dim)

        # build the model
        
        self.dec = UniversalTransformerDecoderLayer(
                    n_layers,
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )

        self.num = 0
        self.num_of_layer_list = th.tensor([0,0,0,0,0,0]).cuda()

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            Ignored. Should always be ``None`` in this version.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = th.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = self.dropout(tensor + self.position_embeddings(positions).expand_as(tensor))

        if (self.act):
            tensor, (remainders, n_updates) = self.act_fn(tensor, input, encoder_mask, self.dec, self.timing_embeddings, self.position_embeddings, self.n_layers, encoder_output)
            self.act_loss = th.mean(remainders + n_updates)
            if self.act_l2:
                tmp_act_l2_loss = None
                for param in self.act_fn.parameters():
                    if tmp_act_l2_loss is None:
                        tmp_act_l2_loss = th.norm(param)
                    else:
                        tmp_act_l2_loss =  tmp_act_l2_loss + th.norm(param)
                self.act_loss = self.act_loss + tmp_act_l2_loss
            """
            #tensor, (remainders, n_updates)            
            n_update = n_updates.reshape(n_updates.shape[0]*n_updates.shape[1])

            self.num += len(n_update)
            for i in range(self.n_layers):
                self.num_of_layer_list[i] += th.sum((n_update == th.tensor([i+1]).float().cuda()).int())
            
            average = 0
            for i in range(self.n_layers):
                average += (i+1) * self.num_of_layer_list[i]
            average /= self.num
            
            variance = 0
            for i in range(self.n_layers):
                variance += ((i+1 - average) * (i+1 - average) * self.num_of_layer_list[i])
            variance /= self.num
            
            print(self.num_of_layer_list)
            print("dec average")
            print(average)
            print("dec variance")
            print(variance)
            """
            
            return tensor, (remainders, n_updates)
            #return tensor, None

        else:
            for i in range(self.n_layers):
                #tensorの形がわかんねえ予想(b, s, emb)
                tensor = tensor + self.position_embeddings(positions).expand_as(tensor)#[s,emb]
                tensor = tensor + self.timing_embeddings(th.tensor([i], device=input.device)).expand_as(tensor)#emb
                tensor = self.dec(tensor, encoder_output, encoder_mask)

            return tensor, None


class UniversalTransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_layers,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask):
        """Forward pass."""
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = th.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        self.variant = variant
        self.n_segments = n_segments
        self.n_positions = n_positions
        self.out_dim = embedding_size

        self.res_net = False
        self.knowledge_split = False
        self.knowledge_compression = False
        self.act_loss = None

        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)
            

        # build the model
        self.layers = nn.ModuleList()
        if self.knowledge_split:
            for _ in range(self.n_layers-1):
                self.layers.append(
                    TransformerEncoderLayer(
                        n_heads,
                        embedding_size,
                        ffn_size,
                        attention_dropout=attention_dropout,
                        relu_dropout=relu_dropout,
                        dropout=dropout,
                        variant=variant,
                        activation=activation,
                    )
                )
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                    knowledge_split=True,
                )
            )
        elif self.knowledge_compression:
            for _ in range(self.n_layers):
                self.layers.append(
                    TransformerEncoderLayer(
                        n_heads,
                        embedding_size,
                        ffn_size,
                        attention_dropout=attention_dropout,
                        relu_dropout=relu_dropout,
                        dropout=dropout,
                        variant=variant,
                        activation=activation,
                    )
                )
            self.layers.append(
                TransformerFFN_in_out_diff(
                    embedding_size,
                    10,
                    ffn_size,
                    relu_dropout=relu_dropout,
                    activation=activation,
                    normalize=True,
                )  
                        
            )
        else:
            for _ in range(self.n_layers):
                self.layers.append(
                    TransformerEncoderLayer(
                        n_heads,
                        embedding_size,
                        ffn_size,
                        attention_dropout=attention_dropout,
                        relu_dropout=relu_dropout,
                        dropout=dropout,
                        variant=variant,
                        activation=activation,
                    )
                )
        self.output_scaling = output_scaling

    def forward(self, input, positions=None, segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param BoolTensor[batch,seqlen] mask:
            The attention mask; 1 means attend, 0 means ignore.
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=th.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = th.zeros_like(input)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)

        if self.res_net:
            res_tensor = tensor
            res_lambda = 0.2
            for i in range(self.n_layers):
                tensor = (1-res_lambda)*tensor + res_lambda*tensor
                tensor = self.layers[i](tensor, mask)
                res_tensor = tensor
        elif self.knowledge_compression:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)
            knowledge_tensor = self.layers[self.n_layers](tensor)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        tensor = tensor * self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean': 
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.knowledge_compression:#追加
            output = tensor
            ret = (output, knowledge_tensor, mask)
            if self.reduction_type == 'none_with_pos_embs':
                ret = (output, mask, position_embs)
            return ret
        elif self.reduction_type is None or 'none' in self.reduction_type:#これbuild_encoderで指定？
            output = tensor
            ret = (output, mask)
            if self.reduction_type == 'none_with_pos_embs':
                ret = (output, mask, position_embs)
            return ret
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )


class TransformerEncoderLayer(nn.Module):
    """Implements a single Transformer encoder layer."""

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
        knowledge_split=False,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.knowledge_split = knowledge_split
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        if self.knowledge_split:
            self.ffn = TransformerFFN_in_out_diff(
            embedding_size,
            int(embedding_size * 3 / 2),
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
            )  
            self.norm2 = LayerNorm(int(embedding_size * 3 / 2), eps=LAYER_NORM_EPS)
        else:
            self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
            )
            self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        """Forward pass."""
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        if self.knowledge_split:
            tensor = self.ffn(tensor)
        else:
            tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout
        self.n_positions = n_positions
        self.out_dim = embedding_size

        self.res_net = False
        self.act_loss = None

        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )
            )

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            Ignored. Should always be ``None`` in this version.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = th.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        if self.res_net:
            res_lambda = 0.2
            res_tensor = tensor
            for layer in self.layers:
                tensor = (1-res_lambda)*tensor + res_lambda*res_tensor
                tensor = layer(tensor, encoder_output, encoder_mask)
                res_tensor = tensor
        else:
            for layer in self.layers:
                tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask):
        """Forward pass."""
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = th.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

class BasicAttention(nn.Module):
    """Implements simple/classical attention."""

    def __init__(self, dim=1, attn='cosine', residual=False, get_weights=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None):
        """ xs: B x query_len x dim
            ys: B x key_len x dim
            TODO: Document this
        """
        bsz = xs.size(0)
        y_len = ys.size(1)
        x_len = xs.size(1)
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = th.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).view(bsz, 1, y_len)
            attn_mask = attn_mask.repeat(1, x_len, 1)
            l1.masked_fill_(attn_mask, -float('inf'))
        l2 = self.softmax(l1)
        lhs_emb = th.bmm(l2, ys)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb.squeeze(self.dim - 1)


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        """Forward pass."""
        # TODO: there are a lot of parameters to document here.

        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    """Implements the FFN part of the transformer."""

    def __init__(self, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        """Forward pass."""
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

class TransformerFFN_in_out_diff(nn.Module):
    """Implements the FFN part of the transformer."""

    def __init__(self, dimin, dimout, dim_hidden, relu_dropout=0, activation='relu', normalize=False):
        super(TransformerFFN_in_out_diff, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.normalize = normalize
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dimin, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dimout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        if self.normalize:
            self.norm = LayerNorm(dimout, eps=LAYER_NORM_EPS)
        # TODO: initialize biases to 0

    def forward(self, x):
        """Forward pass."""
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        if self.normalize:
            x = self.norm(x)
        return x


### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.01
        self.res_net = False

    def forward(self, tensor, inputs, mask, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = th.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        remainders = th.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = th.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_tensor = th.zeros_like(inputs).type(th.FloatTensor).cuda()
        step = 0

        seq_len = inputs.size(1)
        positions = inputs.new(seq_len).long()
        positions = th.arange(seq_len, out=positions).unsqueeze(0)
        # for l in range(self.num_layers):
        if self.res_net:
            res_tensor = tensor
            res_lambda = 0.2
        while( ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            #any() 1つでも0以外があればTrue
            #while(((n_updates < max_hop)).byte().any()):なぜかError
            # Add timing signal
            if self.res_net:
                tensor = (1-res_lambda)*tensor + res_lambda*res_tensor + pos_enc(positions).expand_as(tensor) + time_enc(th.tensor([step], device=inputs.device)).expand_as(tensor)
            else:
                tensor = tensor + pos_enc(positions).expand_as(tensor) + time_enc(th.tensor([step], device=inputs.device)).expand_as(tensor)#emb#[s,emb]

            p = self.sigma(self.p(tensor)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new tensor and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = (p * still_running + new_halted * remainders).cuda()

            #dec
            if(encoder_output is not None):
                tensor = fn(tensor, encoder_output, mask)
            else:
                # apply transformation on the tensor
                tensor = fn(tensor, mask)
                
            if self.res_net:
                res_tensor = tensor

            # update running part in the weighted tensor and keep the rest
            if tensor.size() == previous_tensor.size():
                previous_tensor = (tensor * update_weights.unsqueeze(-1)) + (previous_tensor * (1 - update_weights.unsqueeze(-1)))
            else:
                previous_tensor = (tensor * update_weights.unsqueeze(-1)) + (previous_tensor.reshape(update_weights.unsqueeze(-1).size()) * (1 - update_weights.unsqueeze(-1)))

            ## previous_tensor is actually the new_tensor at end of hte loop 
            ## to save a line I assigned to previous_tensor so in the next 
            ## iteration is correct. Notice that indeed we return previous_tensor
            step+=1
            #print("step")
            #print(step)
        return previous_tensor, (remainders, n_updates)

class ACT_Light(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_Light, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.01
        self.res_net = False

    def forward(self, tensor, inputs, mask, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        halting_probability = th.zeros(inputs.shape[0]).cuda()
        ## [B, S]
        remainders = th.zeros(inputs.shape[0]).cuda()
        ## [B, S]
        n_updates = th.zeros(inputs.shape[0]).cuda()
        ## [B, S, HDD]
        previous_tensor = th.zeros_like(tensor).type(th.FloatTensor).cuda()

        step = 0
        seq_len = inputs.size(1)
        positions = inputs.new(seq_len).long()
        positions = th.arange(seq_len, out=positions).unsqueeze(0)
        # for l in range(self.num_layers):
        if self.res_net:
            res_tensor = tensor
            res_lambda = 0.2
        while( ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            #any() 1つでも0以外があればTrue
            #while(((n_updates < max_hop)).byte().any()):なぜかError
            # Add timing signal
            if self.res_net:
                tensor = (1-res_lambda)*tensor + res_lambda*res_tensor + pos_enc(positions).expand_as(tensor) + time_enc(th.tensor([step], device=inputs.device)).expand_as(tensor)
            else:
                tensor = tensor + pos_enc(positions).expand_as(tensor) + time_enc(th.tensor([step], device=inputs.device)).expand_as(tensor)#emb#[s,emb]

            seq_vec = self.universal_sentence_embedding(tensor, mask)

            p = self.sigma(self.p(seq_vec)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new tensor and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = (p * still_running + new_halted * remainders).cuda()

            #dec
            if(encoder_output is not None):
                tensor = fn(tensor, encoder_output, mask)
            else:
                # apply transformation on the tensor
                tensor = fn(tensor, mask)
                
            if self.res_net:
                res_tensor = tensor


            if tensor.size() == previous_tensor.size():
                previous_tensor = (tensor * update_weights.unsqueeze(-1).unsqueeze(-1)) + (previous_tensor * (1 - update_weights.unsqueeze(-1).unsqueeze(-1)))
            else:
                previous_tensor = (tensor * update_weights) + (previous_tensor * (1 - update_weights))


            # update running part in the weighted tensor and keep the rest
            """
            if tensor.size() == previous_tensor.size():
                previous_tensor = (tensor * update_weights.unsqueeze(-1)) + (previous_tensor * (1 - update_weights.unsqueeze(-1)))
            else:
                previous_tensor = (tensor * update_weights.unsqueeze(-1)) + (previous_tensor.reshape(update_weights.unsqueeze(-1).size()) * (1 - update_weights.unsqueeze(-1)))
            """
            ## previous_tensor is actually the new_tensor at end of hte loop 
            ## to save a line I assigned to previous_tensor so in the next 
            ## iteration is correct. Notice that indeed we return previous_tensor

            #print("step")
            #print(step)
            step+=1
        return previous_tensor, (remainders, n_updates)
    
    def universal_sentence_embedding(self, sentences, mask, sqrt=True, use_mask=True):
        """
        Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).
        This is really just sum / sqrt(len).
        :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
        :param ByteTensor: an N x T binary matrix of paddings
        :return: an N x D matrix of sentence embeddings
        :rtype Tensor:
        """

        # need to mask out the padded chars
        #sentences = [B,L,emb]
        #sentences = sentences.permute(0, 2, 1)

        sentence_sums = (sentences * mask.float().unsqueeze(-1)).squeeze(-1)

        if use_mask:
            divisor = mask.sum(dim=1).view(-1, 1).float()
            if sqrt:
                divisor = divisor.sqrt()
            print(sentence_sums.shape)
            print(divisor.shape)
            sentence_sums = sentence_sums / divisor
        return sentence_sums
