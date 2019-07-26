#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn

from parlai.core.utils import neginf
from projects.my_wizard_of_wikipedia.generator.transformer.modules import TransformerGeneratorModel



def universal_sentence_embedding(sentences, mask, sqrt=True):
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
    sentence_sums = th.bmm(
        sentences.permute(0, 2, 1),
        mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class EndToEndModel(TransformerGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        self.encoder = ContextKnowledgeEncoder(self.encoder)
        self.decoder = ContextKnowledgeDecoder(self.decoder)

    def reorder_encoder_states(self, encoder_out, indices):
        enc, mask, ckattn = encoder_out
        if not th.is_tensor(indices):
            indices = th.LongTensor(indices).to(enc.device)
        enc = th.index_select(enc, 0, indices)
        mask = th.index_select(mask, 0, indices)
        ckattn = th.index_select(ckattn, 0, indices)
        return enc, mask, ckattn


class ContextKnowledgeEncoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # The transformer takes care of most of the work, but other modules
        # expect us to have an embeddings available
        self.embeddings = transformer.embeddings
        self.embed_dim = transformer.embeddings.embedding_dim
        self.transformer = transformer
        self.soft_attention = True
        self.n_use_knowlege = 5 #使う知識数
        self.knowledge_lamda = 1

    def forward(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids):
        # encode the context, pretty basic

        context_encoded, context_mask = self.transformer(src_tokens)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.transformer(know_flat)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)
        know_use /= np.sqrt(self.embed_dim)

        ck_attn = th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))

        #print(use_cs_ids)
        if self.soft_attention:
            # pick the true chosen sentence. remember that TransformerEncoder outputs
            #   (batch, time, embed)
            # but because know_encoded is a flattened, it's really
            #   (N * K, T, D)
            # We need to compute the offsets of the chosen_sentences
            cs_encoded = None
            #print(ck_attn)
            softmax_cs_weight = th.nn.functional.softmax((ck_attn * self.knowledge_lamda), dim=1)
            _, T, D = know_encoded.size()
            know_encoded = know_encoded.reshape((N*K, -1))
            softmax_cs_weight = softmax_cs_weight.reshape(-1,1).expand(N*K, T*D)
            cs_encoded = (know_encoded * softmax_cs_weight).reshape((N,K,T,D)).sum(dim=1)
            cs_mask = know_mask[th.arange(N, device=cs_ids.device) * K] #全部１っぽい
            # finally, concatenate it all
            full_enc = th.cat([cs_encoded, context_encoded], dim=1)
            full_mask = th.cat([cs_mask, context_mask], dim=1)

            # also return the knowledge selection mask for the loss
            return full_enc, full_mask, ck_attn
        
        else:
            if not use_cs_ids:
                # if we're not given the true chosen_sentence (test time), pick our
                # best guess
                # cs_idsが使われるやつ
                _, cs_ids = ck_attn.max(1)
                #_, cs_ids = self.second_max(ck_attn, 1)

            # pick the true chosen sentence. remember that TransformerEncoder outputs
            #   (batch, time, embed)
            # but because know_encoded is a flattened, it's really
            #   (N * K, T, D)
            # We need to compute the offsets of the chosen_sentences
            cs_offsets = th.arange(N, device=cs_ids.device) * K + cs_ids
            cs_encoded = know_encoded[cs_offsets]
            # but padding is (N * K, T)
            cs_mask = know_mask[cs_offsets]

            # finally, concatenate it all
            full_enc = th.cat([cs_encoded, context_encoded], dim=1)
            full_mask = th.cat([cs_mask, context_mask], dim=1)

            # also return the knowledge selection mask for the loss
            return full_enc, full_mask, ck_attn

    def second_max(self, target_tensor, axis):
        #todo make axis != 1 
        #target_tensor (B,N) 
        #return (second_val, second_idx)
        first_idx = 0
        second_idx = 0
        first_tmp = th.tensor(-99.0, device=target_tensor.device)
        second_tmp = th.tensor(-99.0, device=target_tensor.device)
     
     #target_tensor(B,N)
        for i, val in enumerate(target_tensor[0]):

            if first_tmp.data < val.data:
                second_idx = first_idx
                second_tmp = first_tmp
                first_idx = i
                first_tmp = val
            elif second_tmp.data < val.data:
                second_tmp = val
                second_idx = i
        second_idx = th.tensor([second_idx], device=target_tensor.device)
        second_tmp = th.tensor([second_tmp], device=target_tensor.device).float
        return second_tmp, second_idx

    def sort_knowledge(self, target_tensor):
        #類似度の高い順に並んだインデックス番号のTensorリストを返す
        #targettensor 後で使うかもしれんし
        #batch操作B,N
        target_taple_list = [(i, val) for i, val in enumerate(target_tensor[0])]
        self.merge_sort(target_taple_list)
        sorted_target_id = th.tensor([i for i, _ in target_taple_list], device=target_tensor.device)
        sorted_target_value = th.tensor([i for _, i in target_taple_list], device=target_tensor.device)

        return (sorted_target_id, sorted_target_value)

    def merge_sort(self, target_taple_list):
        if(len(target_taple_list) > 1):
            m = int(len(target_taple_list) / 2) 
            #n = int(len(target_taple_list) - m)
            a1 = target_taple_list[:m]
            a2 = target_taple_list[m:]
            self.merge_sort(a1)
            self.merge_sort(a2)
            self.merge(a1, a2, target_taple_list)

    def merge(self, a1, a2, a):
        i = 0
        j = 0
        while(i < len(a1) or j < len(a2)):
            if(j >= len(a2) or (i<len(a1) and a1[i][1] > a2[j][1])):
                a[i+j] = a1[i]
                i += 1
            else:
                a[i+j] = a2[j]
                j += 1


class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder

        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)
