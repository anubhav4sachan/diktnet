import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.elmo import batch_to_ids

from zsdg.enc2dec.base_modules import BaseRNN
from zsdg.enc2dec.decoders import TEACH_FORCE, TEACH_GEN, GEN, Attention
from zsdg.utils import FLOAT, LONG, cast_type


class ElmoDecoderBase(BaseRNN):
    def __init__(self, *args, **kwargs):

        super(ElmoDecoderBase, self).__init__(*args, **kwargs)

    def _batch_ids_to_elmo(self, in_batch):
        ids_flat = in_batch.flatten()
        batch_words = [[self.vocab[id_i]] for id_i in ids_flat]
        elmo_ids = batch_to_ids(batch_words)
        return elmo_ids


class ElmoDecoderRNN(ElmoDecoderBase):
    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id,
                 eos_id, vocab, n_layers=1, rnn_cell='lstm', input_dropout_p=0,
                 dropout_p=0, use_attention=False, attn_mode='cat',
                 attn_size=None, use_gpu=True, embedding=None, output_size=None,
                 tie_output_embed=False):

        super(ElmoDecoderRNN, self).__init__(vocab_size, input_size,
                                             hidden_size, input_dropout_p,
                                             dropout_p, n_layers, rnn_cell, False)

        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.vocab = vocab
        self.use_gpu = use_gpu

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)
        else:
            self.embedding = embedding

        if use_attention:
            self.attention = Attention(self.hidden_size, attn_size, attn_mode,
                                       project=True)

        if tie_output_embed:
            self.project = lambda x: x * self.embedding.weight.transpose(0, 1)
        else:
            self.project = nn.Linear(self.hidden_size, self.output_size)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))
        predicted_softmax = self.function(logits, dim=logits.dim()-1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, batch_size, inputs=None, init_state=None,
                attn_context=None, mode=TEACH_FORCE, gen_type='greedy',
                beam_size=4):
        # sanity checks
        ret_dict = dict()

        if self.use_attention:
            # calculate initial attention
            ret_dict[ElmoDecoderRNN.KEY_ATTN_SCORE] = list()

        if mode == GEN:
            inputs = None

        if gen_type != 'beam':
            beam_size = 1

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            bos_batch = np.ones((batch_size, 1), dtype=np.int32) * self.sos_id
            bos_var = self._batch_ids_to_elmo(bos_batch)
            # bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            decoder_input = cast_type(bos_var, LONG, self.use_gpu)

        if mode == GEN and gen_type == 'beam':
            # if beam search, repeat the initial states of the RNN
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = (self.repeat_state(h, batch_size, beam_size),
                                  self.repeat_state(c, batch_size, beam_size))
            else:
                decoder_hidden = self.repeat_state(init_state,
                                                   batch_size, beam_size)
        else:
            decoder_hidden = init_state

        decoder_outputs = [] # a list of logprob
        sequence_symbols = [] # a list word ids
        back_pointers = [] # a list of parent beam ID
        lengths = np.array([self.max_length] * batch_size * beam_size)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)

            if self.use_attention:
                ret_dict[ElmoDecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == 'beam':
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)
                    seq_score = seq_score[:, 0:self.output_size]
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)

                top_v, top_id = seq_score.topk(beam_size)

                back_ptr = top_id.div(self.output_size).view(-1, 1)
                symbols = top_id.fmod(self.output_size).view(-1, 1)
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, attn_context)

            # in teach forcing mode, we don't need symbols.
            decoder_outputs = decoder_output

        else:
            # do free running here
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(
                    decoder_input, decoder_hidden, attn_context)

                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                symbols_elmo = self._batch_ids_to_elmo(symbols.cpu().numpy())
                decoder_input = cast_type(symbols_elmo, LONG, self.use_gpu)

            decoder_outputs = torch.cat(decoder_outputs, dim=1)

            if gen_type == 'beam':
                # do back tracking here to recover the 1-best according to
                # beam search.
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]

                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)

                    selected_symbols = []
                    selected_parents =[]
                    for b_id in range(batch_size):
                        selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_symbols.append(symbol2ds[b_id, max_seq_id[b_id]])

                    final_seq_symbols.append(torch.cat(selected_symbols).unsqueeze(1))
                    max_seq_id = torch.cat(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]

        # save the decoded sequence symbols and sequence length
        ret_dict[ElmoDecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[ElmoDecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict


class ElmoPointerGen(ElmoDecoderBase):

    def __init__(self,
                 vocab_size,
                 max_len,
                 input_size,
                 hidden_size,
                 sos_id,
                 eos_id,
                 vocab,
                 n_layers=1,
                 rnn_cell='lstm',
                 input_dropout_p=0,
                 dropout_p=0,
                 attn_mode='cat',
                 attn_size=None,
                 use_gpu=True,
                 embedding=None):

        super(ElmoPointerGen, self).__init__(vocab_size,
                                             input_size,
                                             hidden_size,
                                             input_dropout_p,
                                             dropout_p,
                                             n_layers,
                                             rnn_cell,
                                             False)

        self.output_size = vocab_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.use_gpu = use_gpu
        self.attn_size = attn_size
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.output_size, self.input_size)
        else:
            self.embedding = embedding

        self.attention = Attention(self.hidden_size, attn_size, attn_mode,
                                   project=True)

        self.project = nn.Linear(self.hidden_size, self.output_size)
        self.sentinel = nn.Parameter(torch.randn((1, 1, attn_size)), requires_grad=True)
        self.register_parameter('sentinel', self.sentinel)

    def forward_step(self, input_var, hidden, attn_ctxs, attn_words, ctx_embed=None):
        """
        attn_size: number of context to attend
        :param input_var: 
        :param hidden: 
        :param attn_ctxs: batch_size x attn_size+1 x ctx_size. If None, then leave it empty
        :param attn_words: batch_size x attn_size
        :return: 
        """
        # we enable empty attention context
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)
        embedded = self.embedding(input_var)
        if ctx_embed is not None:
            embedded += ctx_embed

        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        if attn_ctxs is None:
            # pointer network here
            logits = self.project(output.contiguous().view(-1, self.hidden_size))
            predicted_softmax = F.log_softmax(logits, dim=1)
            return predicted_softmax, None, hidden, None, None
        else:
            attn_size = attn_words.size(1)
            combined_output, attn = self.attention(output, attn_ctxs)

            # output: batch_size x seq_len x hidden_size
            # attn: batch_size x seq_len x (attn_size+1)

            # pointer network here
            rnn_softmax = F.softmax(self.project(output.contiguous().view(-1, self.hidden_size)), dim=1)
            g = attn[:, :, 0].contiguous()
            ptr_attn = attn[:, :, 1:].contiguous()
            ptr_softmax = Variable(torch.zeros((batch_size * seq_len * attn_size, self.vocab_size)))
            ptr_softmax = cast_type(ptr_softmax, FLOAT, self.use_gpu)

            # convert words and ids into 1D
            flat_attn_words = attn_words.unsqueeze(1).repeat(1, seq_len, 1).view(-1, 1)
            flat_attn = ptr_attn.view(-1, 1)

            # fill in the attention into ptr_softmax
            ptr_softmax = ptr_softmax.scatter_(1, flat_attn_words, flat_attn)
            ptr_softmax = ptr_softmax.view(batch_size * seq_len, attn_size, self.vocab_size)
            ptr_softmax = torch.sum(ptr_softmax, dim=1)

            # mix the softmax from rnn and pointer
            mixture_softmax = rnn_softmax * g.view(-1, 1) + ptr_softmax

            # take the log to get logsoftmax
            logits = torch.log(mixture_softmax.clamp(min=1e-8))
            predicted_softmax = logits.view(batch_size, seq_len, -1)
            ptr_softmax = ptr_softmax.view(batch_size, seq_len, -1)

            return predicted_softmax, ptr_softmax, hidden, ptr_attn, g

    def forward(self, batch_size, attn_context, attn_words,
                inputs=None, init_state=None, mode=TEACH_FORCE,
                gen_type='greedy', ctx_embed=None):

        # sanity checks
        ret_dict = dict()

        if mode == GEN:
            inputs = None

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            bos_batch = np.ones((batch_size, 1), dtype=np.int32) * self.sos_id
            bos_var = self._batch_ids_to_elmo(bos_batch)
            # bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            decoder_input = cast_type(bos_var, LONG, self.use_gpu)

        # append sentinel to the attention
        if attn_context is not None:
            attn_context = torch.cat([self.sentinel.expand(batch_size, 1, self.attn_size),
                                      attn_context], dim=1)

        decoder_hidden = init_state
        decoder_outputs = [] # a list of logprob
        sequence_symbols = [] # a list word ids
        attentions = []
        pointer_gs = []
        pointer_outputs = []
        lengths = np.array([self.max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)

            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            pred_softmax, ptr_softmax, decoder_hidden, attn, step_g = self.forward_step(
                decoder_input, decoder_hidden, attn_context, attn_words, ctx_embed)

            # in teach forcing mode, we don't need symbols.
            attentions = attn
            decoder_outputs = pred_softmax
            pointer_gs = step_g
            pointer_outputs = ptr_softmax

        else:
            # do free running here
            for di in range(self.max_length):
                pred_softmax, ptr_softmax, decoder_hidden, step_attn, step_g = self.forward_step(
                    decoder_input, decoder_hidden, attn_context, attn_words, ctx_embed)

                symbols = decode(di, pred_softmax)

                # append the results into ctx dictionary
                attentions.append(step_attn)
                pointer_gs.append(step_g)
                pointer_outputs.append(ptr_softmax)
                symbols_elmo = self._batch_ids_to_elmo(symbols.cpu().numpy())
                decoder_input = cast_type(symbols_elmo, LONG, self.use_gpu)

            # make list be a tensor
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            pointer_outputs = torch.cat(pointer_outputs, dim=1)
            pointer_gs = torch.cat(pointer_gs, dim=1)

        # save the decoded sequence symbols and sequence length
        ret_dict[self.KEY_ATTN_SCORE] = attentions
        ret_dict[self.KEY_SEQUENCE] = sequence_symbols
        ret_dict[self.KEY_LENGTH] = lengths
        ret_dict[self.KEY_G] = pointer_gs
        ret_dict[self.KEY_PTR_SOFTMAX] = pointer_outputs
        ret_dict[self.KEY_PTR_CTX] = attn_words

        return decoder_outputs, decoder_hidden, ret_dict
