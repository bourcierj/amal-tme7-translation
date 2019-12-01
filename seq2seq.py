"""Defines the Encoder-Decoder Sequence-to-Sequence model"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encoder module.
    Returns the final hidden and cell states, that make the context vector.
    """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=2,
                 dropout=0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,
                           dropout=dropout)

    def forward(self, x, lengths):
        # embed input
        seq_len, batch_size = tuple(x.size())
        embed = self.embedding(x)  # (seq_len, batch, embedding)
        # pack embeddings and pass to the RNN
        packed = pack_padded_sequence(embed, lengths, enforce_sorted=False)
        out, context = self.rnn(packed)  # context is (2, batch, hidden)
        #@todo
        #@bug: the (h_T, c_T) vectors are only for the last time step, however all inputs
        # in the batchs do not have equal lengths: we need to extract, for each input i in the
        # batch of length S, the context vector corresponding to out[S, i, :]

        # return the context vector
        return context


class Decoder(nn.Module):
    """
    Decoder module.
    Does a single step of decoding, that is a single token per time-step
    """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=2,
                 dropout=0):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,
                           dropout=dropout)
        self.lin = nn.Linear(hidden_size, input_size)

    def forward(self, xt, state):
        # input: xt (batch,), state (num_layers, batch, hidden)
        embed = self.embedding(xt)  # (batch, embedding)
        # unsqueeze to add a time dimension
        embed = embed.unsqueeze(0)
        # Note: no need to pack sequences here, only one time step
        out, state = self.rnn(embed, state)  # (1, batch, hidden)
        # decodes output through a linear layer
        out = self.lin(out.squeeze(0))  # (batch, vocab)
        return out, state

class EncoderDecoder(nn.Module):
    """
    Encoder-decoder sequence-to-sequence model (Sutskever et al. 2014)
    (https://arxiv.org/abs/1409.3215).
    """
    def __init__(self, vocab_size_orig, vocab_size_dest, embedding_size, hidden_size,
                 num_layers=2, dropout=0):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(vocab_size_orig, embedding_size, hidden_size, num_layers,
                               dropout)
        self.decoder = Decoder(vocab_size_dest, embedding_size, hidden_size, num_layers,
                               dropout)

    def forward(self, orig, dest, orig_lengths, teacher_forcing_ratio=0):
        # feed origin input into the encoder to receive a context vector
        context = self.encoder(orig, orig_lengths)

        outputs = list()  # stores decorer outputs
        dest_len, batch_size = tuple(dest.size())
        #tensor to store decoder outputs
        outputs = torch.zeros(dest_len, batch_size,
                              self.decoder.input_size).to(dest.device)

        # first input to the decoder is the <sos> token
        in_t = dest[0, :]
        # The decoder loop
        # Note: the 0th element of our outputs tensor remains all zeros (will be cut
        # when computing the loss
        for t in range(1, dest.size(0)):
            out, context = self.decoder(in_t, context)
            outputs[t] = out
            # decide, for this time step, if we use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #if teacher forcing, use ground-truth next token as next input
            #if not, use predicted token
            if teacher_force:
                in_t = dest[t, :]
            else:
                pred = out.argmax(1)  # extract predicted tokens
                in_t = pred

        return outputs


if __name__ == '__main__':

    from translation_data import *

    batch_size = 8
    loader, orig_vocab, dest_vocab = get_dataloader_and_vocabs(batch_size, max_len=10)
    data, lengths, target = next(iter(loader))

    # Test encoder
    print(f"Origin batch size: {tuple(data.size())}, with lengths: {lengths}\n")
    encoder = Encoder(len(orig_vocab), embedding_size=4, hidden_size=6)
    context = encoder(data, lengths)
    print("Encoder output size (context):", tuple(context[0].size()))
    # Test decoder
    decoder = Decoder(len(dest_vocab), embedding_size=4, hidden_size=6)
    target_0 = target[0, :]
    print(f"Destination batch size: {tuple(target.size())}, "
          f"input at time t=0: {tuple(target_0.size())}")
    out, state = decoder(target_0, context)
    print("Decoder output size:", tuple(out.size()), '\n')

    # Test encoder-decoder
    net = EncoderDecoder(len(orig_vocab), len(dest_vocab), embedding_size=4,
                         hidden_size=6)
    out = net(data, target, lengths, teacher_forcing_ratio=0.5)

    print("Encoder-decoder output size:", tuple(out.size()), '\n')


    def test_forward(encoder):
        """
        Test forward encoder.
        Assert that context vector is correct with packed sequences and incorrect
        with padded sequences, with an example with variable sequence lenghts.
        """
        with torch.no_grad():
            # toy tensor
            x = torch.tensor([[1, 1, 1, 1],  # PAD = 0, SOS = 1, EOS = 2
                              [4, 6, 4, 7],
                              [2, 7, 6, 2],
                              [0, 4, 5, 0],
                              [0, 2, 2, 0]])
            lengths = [3, 5, 5, 3]
            seq_len, batch_size = tuple(x.size())
            embed = encoder.embedding(x)
            # first pass with packed sequence
            packed = pack_padded_sequence(embed, lengths, enforce_sorted=False)
            out1, context_1 = encoder.rnn(packed)
            hidden_1, cell_1 = context_1

            # second pass with padded sequence
            out2, context_2 = encoder.rnn(embed)
            hidden_2, cell_2 = context_2

            assert(all(torch.allclose(t_1[:, [1, 2], :], t_2[:, [1, 2], :])
                       for (t_1, t_2) in zip(context_1, context_2)))

            assert(not all(torch.allclose(t_1[:, [0, 3], :], t_2[:, [0, 3], :])
                           for (t_1, t_2) in zip(context_1, context_2)))

    encoder.test_forward()
