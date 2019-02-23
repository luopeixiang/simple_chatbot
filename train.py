import os
import random

import torch

from data import SOS_token
from utils import *

MAX_LENGTH = 10

def train(input_variable, lengths, target_variable, mask,
    max_target_len, encoder, decoder, embedding,
    encoder_optimizer, decoder_optimizer, batch_size, clip,
    max_length=MAX_LENGTH):


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    encoder_outputs,encoder_hidden = encoder(input_variable, lengths)
    #decoder_input = torch.LongTensor([[1]]) * SOS_token
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    #encoder_hidden : (n_layers*n_direction, batch_size, hidden_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    teacher_forcing_rotia = 1
    use_teacher_forcing = True if random.random() < teacher_forcing_rotia else False



    loss = 0
    print_losses = []
    n_totals = 0

    for t in range(max_target_len):

        decoder_out, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
            )

        mask_loss, nTotal = maskNLLLoss(decoder_out, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

        if use_teacher_forcing:
            decoder_input = target_variable[t].view(1, -1)
        else:
            _, decoder_input = torch.max(decoder_out, dim=1)
            decoder_input = decoder_input.view(1, -1)

    loss.backward()
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
    decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
    save_dir, n_iteration, batch_size, print_every, save_every, clip,
    corpus_name, loadFilename):

    #get data
    batches_data = [[random.choice(pairs) for i in range(batch_size)]
                        for _ in range(n_iteration)]

    print_loss = 0
    for iteration, batch_data in enumerate(batches_data, 1):
        input_var, input_lengths, target_var, mask, max_target_len = \
            batch2TrainData(batch_data, voc)
        loss = train(input_var, input_lengths, target_var, mask,
            max_target_len, encoder, decoder, embedding,
            encoder_optimizer, decoder_optimizer, batch_size, clip,
            max_length=MAX_LENGTH)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("iteration: {}/{} Loss: {:.4f}".format(iteration+1, n_iteration, print_loss_avg))
            print_loss = 0

        #save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(
                    save_dir,
                    model_name,
                    '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, encoder.hidden_size)
                    )
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
