#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn

from parlai.core.utils import neginf
from projects.my_wizard_of_wikipedia.generator.transformer.transformer_universal_modules import TransformerGeneratorModel



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
        self.soft_attention = False
        #self.n_use_knowlege = 5 #使う知識数
        self.knowledge_lamda = 1
        self.knowledge_split = True
        self.ck_mask =None
        self.know_tokens = None
        self.cs_ids = None
        self.use_cs_ids =None

    def forward(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids):
        # encode the context, pretty basic
        #N:バッチサイズ, K:知識数, T:時間, D:埋め込みサイズ, Tk:
        #src_tokens torch.Size([B, T])
        #cs_ids tensor([0, 0, 0, 0], device='cuda:0')
        #use_cs_ids trainならTrue

        self.know_tokens = know_tokens
        self.ck_mask = ck_mask 
        self.cs_ids = cs_ids
        self.use_cs_ids = use_cs_ids

        context_encoded, context_mask = self.transformer(src_tokens)

        if self.knowledge_split:
                        # make all the knowledge into a 2D matrix to encode
            N, K, Tk = know_tokens.size()
            know_encoded, know_mask = self.transformer(know_tokens.reshape(-1, Tk))

            # compute our sentence embeddings for context and knowledge
            context_use = universal_sentence_embedding(context_encoded[:,:,int(self.embed_dim/2)], context_mask)
            know_use = universal_sentence_embedding(know_encoded[:,:,int(self.embed_dim/2)], know_mask)
            
            context_encoded = context_encoded[:,:,-self.embed_dim]
            know_encoded = know_encoded[:,:,-self.embed_dim]

                        # remash it back into the shape we need
            know_use = know_use.reshape(N, know_tokens.size(1), int(self.embed_dim/2)) / np.sqrt(self.embed_dim/2)
            context_use = text_use / np.sqrt(self.embed_dim/2)
        else:
            # make all the knowledge into a 2D matrix to encode
            N, K, Tk = know_tokens.size()
            know_encoded, know_mask = self.transformer(know_tokens.reshape(-1, Tk))

            # compute our sentence embeddings for context and knowledge
            context_use = universal_sentence_embedding(context_encoded, context_mask)
            know_use = universal_sentence_embedding(know_encoded, know_mask)

            # remash it back into the shape we need
            know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim) / np.sqrt(self.embed_dim)
            context_use = text_use / np.sqrt(self.embed_dim)

        ck_attn = th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf

        ck_attn.masked_fill_(ck_mask==0, neginf(context_encoded.dtype))
        if self.soft_attention:
            # pick the true chosen sentence. remember that TransformerEncoder outputs
            #   (batch, time, embed)
            # but because know_encoded is a flattened, it's really
            #   (N * K, T, D)
            # We need to compute the offsets of the chosen_sentences
            cs_encoded = None
            softmax_cs_weight = th.nn.functional.softmax((ck_attn * self.knowledge_lamda), dim=1)
            #add
            true_ids_weight = th.zeros(softmax_cs_weight.shape, device=softmax_cs_weight.device, dtype=softmax_cs_weight.dtype)
            for temp in true_ids_weight:
                temp[0] = 1

            weight_abs = th.abs(softmax_cs_weight - true_ids_weight)
            weight_abs *= weight_abs
            _, T, D = know_encoded.size()
            # finally, concatenate it all
            
            know_encoded.masked_fill_(know_mask==0, 0)
            
            full_enc = th.cat([(know_encoded.reshape((N*K, -1)) * th.nn.functional.softmax((ck_attn * self.knowledge_lamda), dim=1).reshape(-1,1).expand(N*K, T*D)).reshape((N,K,T,D)).sum(dim=1), context_encoded], dim=1)
            full_mask = th.cat([know_mask[th.arange(N, device=cs_ids.device) * K], context_mask], dim=1)

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
    
    def output_choose_knowledge(self, out_tokens):
        #outputと知識をsoftmaxして正解知識を選べるか
        # encode the context, pretty basic
        #N:バッチサイズ, K:知識数, T:時間, D:埋め込みサイズ, Tk:
        context_encoded, context_mask = self.transformer(out_tokens)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = self.know_tokens.size()
        know_encoded, know_mask = self.transformer(self.know_tokens.reshape(-1, Tk))

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, self.know_tokens.size(1), self.embed_dim) / np.sqrt(self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)

        ck_attn = th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        #~はInvert-2^(N-1) to 2^(N-1)-1
       
        ck_attn.masked_fill_(self.ck_mask==0, neginf(context_encoded.dtype))

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_encoded = None
        softmax_cs_weight = th.nn.functional.softmax((ck_attn * self.knowledge_lamda), dim=1)
        """
        #cs_idは0 softmax_cs_weightは(B,knowledge)
        true_ids_weight = th.zeros(softmax_cs_weight.shape, device=softmax_cs_weight.device, dtype=softmax_cs_weight.dtype)
        for temp in true_ids_weight:
            temp[0] = 1

        loss = softmax_cs_weight - true_ids_weight
        loss = loss * loss 
        loss[loss == 0] = 0.000001
        loss = th.sqrt(loss)
        loss = th.sum(loss) / N
        #print(loss)

        self.know_tokens = None
        self.ck_mask = None
        self.cs_ids = None
        self.use_cs_ids = None
        # also return the knowledge selection mask for the loss
        """
        return softmax_cs_weight


class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder

        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)
