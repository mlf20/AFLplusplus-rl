#!/usr/bin/env python3
# encoding: utf-8
"""
Example Python Module for AFLFuzz

@author:     Christian Holler (:decoder)

@license:

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

@contact:    choller@mozilla.com
"""

# General Imports
import random
import os


from torch import exp, clamp, min, max
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import ChainMap, deque
from subprocess import Popen
from subprocess import PIPE
import torch
import numpy as np
from copy import copy
import re
from multiprocessing import shared_memory
import os
import torch.multiprocessing as mp
import threading
import queue




input_dimension=1
recurrent=False
hidden_size=64
weight_size=512
l=1e-4

deep_bandit = None
rollouts = None
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


optimiser = None # optim.Adam(deep_bandit.parameters())



current_context = torch.zeros((1000, 1)).to(device)
mapping ={}

current_context = None
SAVE_DIR            = f'logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}_PPO/'



reward =  -1
total_edges  = None
exp_update_step = 20
edges_covered = set()
cyclomatic_complexity = 1
prev_max_cyclomatic_complexity = 1
current_regularization = None
current_log_var = None
current_mean = None
max_cyclomatic_complexity = 1
_timeout = 10
status = None
step_count = 0
max_reward = 0
program = ''

PREV_VIRGIN_BITS    = 0
PREV_TOTAL_CRASHES  = 0
TOTAL_EXECUTIONS    = 0

TOTAL_STEP_COUNTER  = 0
SAVE_FREQ           = 5000


class ConcreteDropout(nn.Module):
    def __init__(self, shape,weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)


        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1
        unif_noise = torch.rand_like(x)






        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x
class QBandit(nn.Module):
    def __init__(self, input_dimension=1, recurrent=False, hidden_size=64, weight_size=512,
                 l=1e-4):
        super(QBandit, self).__init__()


        weight_regularizer = l**2. / input_dimension
        dropout_regularizer = 2. / input_dimension

        self.recurrent_hidden_state_size = hidden_size
        self._hidden_size = hidden_size
        self.global_embeddings = nn.Linear(input_dimension, hidden_size)


        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)
        self.vt = nn.Linear(weight_size, 1, bias=False)
        self.linear4_mu = nn.Linear(weight_size, 1, bias=False)
        self.linear4_logvar = nn.Linear(weight_size, 1, bias=False)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.dec = nn.GRUCell(hidden_size, hidden_size)

        self.conc_drop1 = ConcreteDropout(shape=(1, 1000, self._hidden_size),
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(shape=(1000, self._hidden_size),
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(shape=(1, 1000, weight_size),
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(shape=(1, 1000, weight_size),
                                             weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(shape=(1, 1000, weight_size),
                                                weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer)

        self.train()



    def forward(self, inputs):
        if type(inputs) == list:
            inputs = torch.cat(inputs)
        x = inputs
        batch_size = x.size(0) if x.dim() > 1 else 1
        if inputs.dim() > 1:
            global_embeds = self.global_embeddings(inputs.type(torch.float32))
            global_embeds = global_embeds.view(-1, global_embeds.shape[0], self._hidden_size)
        else:
            global_embeds = self.global_embeddings(inputs.type(torch.float32))
            global_embeds = global_embeds.view(-1, 1, self._hidden_size)

        regularization = torch.empty(5, device=global_embeds.device)
        enc_states, rnn_hxs = self.gru(global_embeds)
        decoder_input = rnn_hxs[-1]
        hidden = torch.zeros([batch_size, self._hidden_size]).to(global_embeds.device)
        hidden = self.dec(decoder_input, hidden)
        blend1, regularization[0] = self.conc_drop1(enc_states,self.W1)
        blend2, regularization[1] = self.conc_drop2(hidden, self.W2)
        blend_sum = torch.tanh(blend1 + blend2)

        out, regularization[2] = self.conc_drop3(blend_sum, self.vt)

        mean, regularization[3] = self.conc_drop_mu(blend_sum, self.linear4_mu)
        log_var, regularization[4] = self.conc_drop_logvar(blend_sum, self.linear4_logvar)
        mean = mean.squeeze()
        out = out.squeeze()
        log_var = log_var.squeeze()
        location = torch.argmax(out)
        mean = mean[location].unsqueeze(0).unsqueeze(0)
        log_var = log_var[location].unsqueeze(0).unsqueeze(0)


        return torch.argmax(out[:-1]), mean, log_var, regularization


    def _calculate_loss(self, reward, mean, log_var, regularisation):
        # discount rewards
        rewards = []
        R = 0
        for r in reward[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)


        reward = torch.tensor(rewards).to(device).squeeze()


        precision = torch.exp(-log_var)
        loss = torch.mean(torch.sum(precision * (reward - mean) ** 2 + log_var), 0) + regularisation.sum()
        return loss




def init(seed):
    """
    Called once when AFLFuzz starts up. Used to seed our RNG.

    @type seed: int
    @param seed: A 32-bit random value
    """

    global deep_bandit
    global rollouts
    global device
    global optimiser

    deep_bandit =  QBandit(input_dimension=input_dimension,
                            recurrent=recurrent,
                            hidden_size=hidden_size,
                            weight_size=weight_size,
                            l=l)
    deep_bandit.to(device)
    rollouts = deque(maxlen=128)

    print('INIT STARTED')


def deinit():
    global SAVE_DIR

    save_path = os.path.abspath(os.getcwd() + f'/{SAVE_DIR}')
    try:
        os.makedirs(save_path)
    except OSError:
        print(save_path)
    print(save_path)

    torch.save(deep_bandit, os.path.join(save_path, 'final_model'+ ".pt"))




def fuzz(buf, add_buf, max_size):
    """
    Called per fuzzing iteration.

    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @type add_buf: bytearray
    @param add_buf: A second buffer that can be used as mutation source.

    @type max_size: int
    @param max_size: Maximum size of the mutated output. The mutation must not
        produce data larger than max_size.

    @rtype: bytearray
    @return: A new bytearray containing the mutated data
    """
    ret = bytearray(100)
    print(GLOBAL_ARRAY)
    GLOBAL_ARRAY.append(1)
    #ret[:3] = random.choice(COMMANDS)
    ret = bytearray(random.randint(0, MAX_ACTIONS))

    return ret

def havoc_mutation(buf, max_size):
    '''
    Perform a single custom mutation on a given input.

    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @type max_size: int
    @param max_size: Maximum size of the mutated output. The mutation must not
        produce data larger than max_size.

    @rtype: bytearray
    @return: A new bytearray containing the mutated data
    '''

    return mutated_buf

def havoc_mutation_probability():
    '''
    Called for each `havoc_mutation`. Return the probability (in percentage)
    that `havoc_mutation` is called in havoc. Be default it is 6%.

    @rtype: int
    @return: The probability (0-100)
    '''
    global MAX_ACTIONS
    prob = random.randint(0, MAX_ACTIONS)
    return prob


def get_action(context):
    global deep_bandit
    return deep_bandit(context)

def update_bandit(minibatch):
    global deep_bandit
    global optimiser
    y, mean, log_var, regularisation = zip(*minibatch)
    y = torch.tensor(y).to(device).squeeze()
    mean = torch.cat(mean).to(device).squeeze()
    log_var = torch.cat(log_var).to(device).squeeze()
    regularisation = torch.stack(regularisation).to(device)
    loss = deep_bandit._calculate_loss(y, mean, log_var, regularisation)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return loss.item()

def update_bitmap_size(size):
    global total_edges
    total_edges = size
    print('BITMAP SIZE UPDATED')

def havoc_mutation_location(buf, havoc_mutation):
    '''
    Called for each `havoc_mutation`.
    For a given buffer return the mutation action to be taken by AFL.
    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @rtype: int
    @return: The action (0-26)
    '''
    print('pick location')
    global step_count
    global exp_update_step
    global rollouts
    global mapping
    global status
    global current_mean
    global current_log_var
    global current_regularization
    global TOTAL_STEP_COUNTER

    step_count += 1
    TOTAL_STEP_COUNTER += 1

    int_list = [[int(str(hex(x)), 16)] for x in list(buf)]
    int_list.append([havoc_mutation])
    current_context = torch.tensor(int_list)
    location_action, current_mean, current_log_var, current_regularization = get_action(current_context)
    status = 'action'
    #print(location_action)
    return int(location_action)






def havoc_mutation_reset():
    '''
    Called for each `havoc_mutation`. Reset the python
    '''
    global step_count
    global TOTAL_STEP_COUNTER
    global SAVE_FREQ
    global SAVE_DIR
    global deep_bandit
    global rollouts
    if len(rollouts) > 0:
        print('updating...')
        __import__("IPython").embed()
        bandit_loss = update_bandit(rollouts)
    rollouts = deque(maxlen=128)

    step_count = 0
    if TOTAL_STEP_COUNTER % SAVE_FREQ == 0:
        save_path = os.path.abspath(os.getcwd() + f'/{SAVE_DIR}')
        try:
            os.makedirs(save_path)
        except OSError:
            print(save_path)
        print(save_path)

        torch.save(deep_bandit, os.path.join(save_path, 'model'+ ".pt"))
    print('UPDATED')

def havoc_mutation_reward(total_crashes, virgin_bits):
    '''
    Called for each `havoc_mutation`. pass vars for computing the reward function
    '''

    global GAE_LAMBDA
    global GAMMA
    global USE_GAE
    global USE_PROPER_TIME_LIMITS
    global TF_WRITER
    global TOTAL_STEP_COUNTER
    global PREV_VIRGIN_BITS
    global PREV_TOTAL_CRASHES
    global TOTAL_EXECUTIONS
    global total_edges
    #virgin_bits = [int(str(hex(x)), 16) for x in list(virgin_bits)][0]
    #print(total_crashes)

    #total_crashes = [int(str(hex(x)), 16) for x in list(total_crashes)][0]
    #print(virgin_bits)
    #print(total_crashes)

    # Compute reward
    '''if total_crashes > PREV_TOTAL_CRASHES: # New crash found
        reward = 3
        #int_list = [int(str(hex(x)), 16) for x in list(buf)]
        PREV_VIRGIN_BITS = virgin_bits
        PREV_TOTAL_CRASHES = total_crashes

    el'''
    if int(virgin_bits) > int(PREV_VIRGIN_BITS): # New bits found compared to previous
        reward = 1
        reward += (int(virgin_bits) - int(PREV_VIRGIN_BITS)) /  total_edges
        #int_list = [int(str(hex(x)), 16) for x in list(buf)]
        PREV_VIRGIN_BITS = virgin_bits
    else:
        reward = -1

    rollouts.append([torch.tensor([copy(reward)]), copy(current_mean), copy(current_log_var), copy(current_regularization)])

    print(reward, virgin_bits, PREV_VIRGIN_BITS, type(virgin_bits), type(PREV_VIRGIN_BITS))

    # Update the last transition with correct reward and done




def introspection():
    string = ''
    return string

if __name__ == '__main__':
    init(3)
    for i in range(10):
        testbyte = bytearray([1, 2, 3, 4])
        action = havoc_mutation_action(testbyte)
        print(f"action: {action}")
        print(f"step counter: {STEP_COUNTER}")
        havoc_mutation_reward(0,0)
        havoc_mutation_reset()


# actions (25 possible actions):
# flip single bit
# set interesting byte value
# set word (2 bytes) to interesting value, little endian.
# Set word to interesting value, big endian.
# Set dword to interesting value, big endian.
# Randomly subtract from byte.
# Randomly add to byte.
# Randomly subtract from word, little endian
# Randomly subtract from word, big endian
# Randomly add to word, little endian

# Randomly add to word, big endian
# Randomly subtract from dword, little endian
# Randomly subtract from dword, big endian
# Randomly add to dword, little endian
# Randomly add to dword, big endian.
# Just set a random byte to a random value. Because, why not. We use XOR with 1-255 to eliminate the possibility of a no-op.
# Clone bytes
# Insert a block of constant bytes (25%).
# Overwrite bytes with a randomly selected chunk bytes.
# Overwrite bytes with fixed bytes.

# Increase byte by 1.
# Decrease byte by 1.
# Flip byte.
# Switch bytes.
# Delete bytes
# Do nothing


# Uncomment and implement the following methods if you want to use a custom
# trimming algorithm. See also the documentation for a better API description.

# def init_trim(buf):
#     '''
#     Called per trimming iteration.
#
#     @type buf: bytearray
#     @param buf: The buffer that should be trimmed.
#
#     @rtype: int
#     @return: The maximum number of trimming steps.
#     '''
#     global ...
#
#     # Initialize global variables
#
#     # Figure out how many trimming steps are possible.
#     # If this is not possible for your trimming, you can
#     # return 1 instead and always return 0 in post_trim
#     # until you are done (then you return 1).
#
#     return steps
#
# def trim():
#     '''
#     Called per trimming iteration.
#
#     @rtype: bytearray
#     @return: A new bytearray containing the trimmed data.
#     '''
#     global ...
#
#     # Implement the actual trimming here
#
#     return bytearray(...)
#
# def post_trim(success):
#     '''
#     Called after each trimming operation.
#
#     @type success: bool
#     @param success: Indicates if the last trim operation was successful.
#
#     @rtype: int
#     @return: The next trim index (0 to max number of steps) where max
#              number of steps indicates the trimming is done.
#     '''
#     global ...
#
#     if not success:
#         # Restore last known successful input, determine next index
#     else:
#         # Just determine the next index, based on what was successfully
#         # removed in the last step
#
#     return next_index
#
# def post_process(buf):
#     '''
#     Called just before the execution to write the test case in the format
#     expected by the target
#
#     @type buf: bytearray
#     @param buf: The buffer containing the test case to be executed
#
#     @rtype: bytearray
#     @return: The buffer containing the test case after
#     '''
#     return buf
#

# def queue_get(filename):
#     '''
#     Called at the beginning of each fuzz iteration to determine whether the
#     test case should be fuzzed
#
#     @type filename: str
#     @param filename: File name of the test case in the current queue entry
#
#     @rtype: bool
#     @return: Return True if the custom mutator decides to fuzz the test case,
#         and False otherwise
#     '''
#     return True
#
# def queue_new_entry(filename_new_queue, filename_orig_queue):
#     '''
#     Called after adding a new test case to the queue
#
#     @type filename_new_queue: str
#     @param filename_new_queue: File name of the new queue entry
#
#     @type filename_orig_queue: str
#     @param filename_orig_queue: File name of the original queue entry
#     '''
#     pass
