'''
Author: Ji-Sung Kim

Some code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.

Code was built while significantly referencing public examples from the
Keras documentation on Github:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python preprocess.py
'''
from __future__ import print_function
import sys

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, izip_longest
import pygame, copy, sys, pdb, math
from grammar import *

from keras.layers.recurrent import LSTM
import numpy as np
import random

import lstm
import preprocess as pp

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Round down num to the nearest multiple of mult. '''
def __roundDown(num, mult):
    return (float(num) - (float(num) % mult))

''' Round up num to nearest multiple of mult. '''
def __roundUp(num, mult):
    return __roundDown(num, mult) + mult

''' Based if upDown < 0 or upDown >= 0, rounds number down or up
    respectively to nearest multiple of mult. '''
def __roundUpDown(num, mult, upDown):
    if upDown < 0:
        return __roundDown(num, mult)
    else:
        return __roundUp(num, mult)

''' From recipes: iterate over list in chunks of n length. '''
def __grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def __sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

#---------------------------------SCRIPT---------------------------------------#
# settings
diversity = 0.5
maxlen = 20
N_epochs = int(sys.argv[1])
fn = 'midi/' + 'original_metheny.mid' # 'And Then I Knew' by Pat Metheny 
if N_epochs != 1:
    outfn = 'midi/' 'deepjazz_on_metheny...' + str(N_epochs) +  '_epochs.midi'
else:
    outfn = 'midi/' 'deepjazz_on_metheny...' + '1_epoch.midi'

# get data, build model
corpus, values, val_indices, indices_val, allMeasures_chords, \
    abstractGrammars = pp.get_data(fn)
print('corpus length:', len(corpus))
print('total # of values:', len(values))

model = lstm.build_model(corpus=corpus, values=values, maxlen=maxlen, 
    N_epochs=N_epochs)

genStream = stream.Stream() # output stream
play = lambda x: midi.realtime.StreamPlayer(x).play()
stop = lambda: pygame.mixer.music.stop()

currOffset = 0
loopEnd = len(allMeasures_chords)
for loopIndex in range(1, loopEnd):
    # generate chords from file
    m1_chords = stream.Voice()
    for j in allMeasures_chords[loopIndex]:
        m1_chords.insert((j.offset % 4), j)

    # generate grammar
    start_index = random.randint(0, len(corpus) - maxlen - 1)
    sentence = corpus[start_index: start_index + maxlen]

    m1_grammar = ''
    running_length = 0.0
    while running_length <= 4.1: # from avg in input file somewhat arbitrary
        x = np.zeros((1, maxlen, len(values)))

        # transform 
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = __sample(preds, diversity)
        next_val = indices_val[next_index]

        # fix first element => must not have < > and not be rest
        if (running_length < 0.00001):
            terms = next_val.split(',')
            tries = 0
            while (terms[0] == 'R' or len(terms) != 2):
                # give up after 1000 tries; choose in a guided, random fashion
                if tries >= 1000:
                    print('Gave up on first note generation')
                    rand = random.randint(0, len(abstractGrammars) - 1)
                    next_val = abstractGrammars[rand].split(' ')[0]

                else:
                    preds = model.predict(x, verbose=0)[0]
                    next_index = __sample(preds, diversity)
                    next_val = indices_val[next_index]

                terms = next_val.split(',')
                tries += 1

        sentence = sentence[1:] 
        sentence.append(next_val)

        # except for first case, add a ' ' separator
        if (running_length > 0.00001): m1_grammar += ' '
        m1_grammar += next_val

        length = float(next_val.split(',')[1])
        running_length += length

    m1_grammar = m1_grammar.replace(' A',' C').replace(' X',' C') \

    # Pruning #1: 'Smooth' the measure, or make sure that everything is in 
    # standard note lengths (0.125, 0.250, 0.333 ... nothing like .482).
    m1_grammar = m1_grammar.split(' ')
    for ix, gram in enumerate(m1_grammar):
        terms = gram.split(',')
        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, random.choice([-1, 1])))
        m1_grammar[ix] = ','.join(terms)
    m1_grammar = ' '.join(m1_grammar)   

    # Get notes from grammar and chords
    m1_notes = unparseGrammar(m1_grammar, m1_chords)

    # fix note offset problems, i.e. same offset so pruning too many.
    # remember - later you can remove 'if (n2.offset - n1.offset) < 0.125' since
    # already adjusted the note durations to be regular enough.

    # Pruning #2: remove repeated notes, and notes WAY too close together.
    for n1, n2 in __grouper(m1_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        # if (n2.offset - n1.offset) < 0.125:
        #     if random.choice(([True] * 10 + [False] * (loopIndex)**2)):
        #         m1_notes.remove(n2)
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                m1_notes.remove(n2)

    # Quality assurance.
    removeIxs = []
    for ix, m in enumerate(m1_notes):
        # QA: make sure nothing is of 0 quarter note length - else changes its len.
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA: make sure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(m1_notes) - 1)):
            if (m.offset == m1_notes[ix + 1].offset and
                isinstance(m1_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    m1_notes = [i for ix, i in enumerate(m1_notes) if ix not in removeIxs]

    # QA: print number of notes in m1_notes. Should see general increasing trend.
    print('After pruning: %s notes' % (len([i for i in m1_notes
        if isinstance(i, note.Note)])))

    # after quality assurance
    for m in m1_notes:
        genStream.insert(currOffset + m.offset, m)
    for mc in m1_chords:
        genStream.insert(currOffset + mc.offset, mc)

    currOffset += 4.0

genStream.insert(0.0, tempo.MetronomeMark(number=130))

# Play the final stream through output
# play is defined as a lambda function above
play(genStream)

# save stream
mf = midi.translate.streamToMidiFile(genStream)
mf.open(outfn, 'wb')
mf.write()
mf.close()