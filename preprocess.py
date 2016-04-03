'''
Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python preprocess.py

'''

from __future__ import print_function

# Definitions
head = lambda x: x[0:5] # from R
tail = lambda x: x[:-6:-1] # from R
ppr = lambda n: '%.3f   %s, %.3f' % (n.offset, n, n.quarterLength) # pretty print note + offset
ppn = lambda n: '%.3f   %s, %.3f' % (n.offset, n.nameWithOctave, n.quarterLength)
trc = lambda s: '%s ...' % (s[0:10]) # pretty print first few chars of str

''' Round down num to the nearest multiple of mult. '''
def roundDown(num, mult):
    return (float(num) - (float(num) % mult))

''' Round up num to nearest multiple of mult. '''
def roundUp(num, mult):
    return roundDown(num, mult) + mult

''' Based if upDown < 0 or upDown >= 0, rounds number down or up
    respectively to nearest multiple of mult. '''
def roundUpDown(num, mult, upDown):
    if upDown < 0:
        return roundDown(num, mult)
    else:
        return roundDown(num, mult)

''' Print a formatted note or rest. '''
def pretty(element):
    if isinstance(element, note.Note):
        return ppn(element)
    else:
        return ppr(element)

''' Pretty print a stream of notes/rests: '''
def prettyPrint(notes):
    for i in notes: print(pretty(i))

''' Print list and stuff. '''
def compareGen(m1_grammar, m1_elements):
    for i, j in zip(m1_grammar.split(' '), m1_elements):
        if isinstance(j, note.Note):
            print('%s  |  %s' % (ppn(j), i))
        else:
            print('%s  |  %s' % (ppr(j), i))

''' From recipes: iterate over list in chunks of n length. '''
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

# Imports
from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, izip_longest
import pygame, copy, sys, pdb, math

# My imports
sys.path.append('./extract')
sys.path.append('./grammar')
from grammar import *

# Parse the MIDI data for separate melody and accompaniment parts.
play = lambda x: midi.realtime.StreamPlayer(x).play()
stop = lambda: pygame.mixer.music.stop()

metheny = converter.parse('andtheniknew_metheny.mid')

# Get melody part, compress into single voice.
# For Metheny piece, Melody is Part #5.
melodyStream = metheny[5]
melody1, melody2 = melodyStream.getElementsByClass(stream.Voice)
for j in melody2:
    melody1.insert(j.offset, j)
melodyVoice = melody1

for i in melodyVoice:
    if i.quarterLength == 0.0:
        i.quarterLength = 0.25

# Change key signature to adhere to compStream (1 sharp, mode = major).
# Also add Electric Guitar. 
melodyVoice.insert(0, instrument.ElectricGuitar())
melodyVoice.insert(0, key.KeySignature(sharps=1, mode='major'))

# The accompaniment parts. Take only the best subset of parts from
# the original data. Maybe add more parts, hand-add valid instruments.
# Should add least add a string part (for sparse solos).
# Verified are good parts: 0, 1, 6, 7 '''
partIndices = [0, 1, 6, 7]
compStream = stream.Voice()
compStream.append([j.flat for i, j in enumerate(metheny) if i in partIndices])

# Full stream containing both the melody and the accompaniment. 
# All parts are flattened. 
fullStream = stream.Voice()
for i in xrange(len(compStream)):
    fullStream.append(compStream[i])
fullStream.append(melodyVoice)

# Extract solo stream, assuming you know the positions ..ByOffset(i, j).
# Note that for different instruments (with stream.flat), you NEED to use
# stream.Part(), not stream.Voice().
# Accompanied solo is in range [478, 548)
soloStream = stream.Voice()
for part in fullStream:
    newPart = stream.Part()
    newPart.append(part.getElementsByClass(instrument.Instrument))
    newPart.append(part.getElementsByClass(tempo.MetronomeMark))
    newPart.append(part.getElementsByClass(key.KeySignature))
    newPart.append(part.getElementsByClass(meter.TimeSignature))
    newPart.append(part.getElementsByOffset(476, 548, includeEndBoundary=True))
    np = newPart.flat
    soloStream.insert(np)

# MELODY: Group by measure so you can classify. 
# Note that measure 0 is for the time signature, metronome, etc. which have
# an offset of 0.0.
melodyStream = soloStream[-1]
allMeasures = OrderedDict()
offsetTuples = [(int(n.offset / 4), n) for n in melodyStream]
measureNum = 0 # for now, don't use real m. nums (119, 120)
for key, group in groupby(offsetTuples, lambda x: x[0]):
    allMeasures[measureNum] = [n[1] for n in group]
    measureNum += 1

# Just play the chord accompaniment with the melody. Refine later. 
# Think I successfully extracted just the chords in chordStream().

# Get the stream of chords.
# offsetTuples_chords: group chords by measure number.
chordStream = soloStream[0]
chordStream.removeByClass(note.Rest)
chordStream.removeByClass(note.Note)
offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

# Generate the chord structure. Use just track 1 (piano) since it is
# the only instrument that has chords. Later, if you have time, you can
# mod this so it works with any MIDI file.
# Group into 4s, just like before. '''
allMeasures_chords = OrderedDict()
measureNum = 0
for key, group in groupby(offsetTuples_chords, lambda x: x[0]):
    allMeasures_chords[measureNum] = [n[1] for n in group]
    measureNum += 1

# Fix for the below problem.
#   1) Find out why len(allMeasures) != len(allMeasures_chords).
#   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
#           actually show up, while the accompaniment's beat 1 right after does.
#           Actually on second thought: melody/comp start on Ab, and resolve to
#           the same key (Ab) so could actually just cut out last measure to loop.
#           Decided: just cut out the last measure. '''
del allMeasures_chords[len(allMeasures_chords) - 1]
assert len(allMeasures_chords) == len(allMeasures)

# TRAINING DATA
abstractGrammars = []
for ix in xrange(1, len(allMeasures)):
    m = stream.Voice()
    for i in allMeasures[ix]:
        m.insert(i.offset, i)
    c = stream.Voice()
    for j in allMeasures_chords[ix]:
        c.insert(j.offset, j)
    parsed = parseMelody(m, c)
    abstractGrammars.append(parsed)

#-----------------------------------MYSTUFF------------------------------------#
corpus = [x for sublist in abstractGrammars for x in sublist.split(' ')]

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

print('corpus length:', len(corpus))
values = set(corpus)
print('total # of values:', len(values))
val_indices = dict((v, i) for i, v in enumerate(values))
indices_val = dict((i, v) for i, v in enumerate(values))

# cut the corpus in semi-redundant sequences of maxlen values
maxlen = 20
step = 3
sentences = []
next_values = []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i: i + maxlen])
    next_values.append(corpus[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(values)), dtype=np.bool)
y = np.zeros((len(sentences), len(values)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, val in enumerate(sentence):
        X[i, t, val_indices[val]] = 1
    y[i, val_indices[next_values[i]]] = 1

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(values))))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(values)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

N_epochs = 128
model.fit(X, y, batch_size=128, nb_epoch=N_epochs)

# save model
json_str = model.to_json()
with open('model_arch.json', 'w') as architecture_f:
    architecture_f.write(json_str)
model.save_weights('model_weights.h5', overwrite=True)


#----------------------------------GENERATE------------------------------------#
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# output
genStream = stream.Stream()

# Where to start, end loop
currOffset = 0
loopEnd = len(allMeasures_chords)

diversity = 0.5

for loopIndex in range(1, loopEnd): # prev: len(allMeasures_chords)
    # generate chords from file
    m1_chords = stream.Voice() # initialize
    for j in allMeasures_chords[loopIndex]:
        m1_chords.insert((j.offset % 4), j)

    # generate grammar
    start_index = random.randint(0, len(corpus) - maxlen - 1)
    sentence = corpus[start_index: start_index + maxlen]
    print('----- Generating with seed: ')
    print(sentence)
    print('-' * 5)

    m1_grammar = ''
    running_length = 0.0
    while running_length <= 4.1: # avg of lengths, somewhat arbitrary
        x = np.zeros((1, maxlen, len(values)))

        # transform 
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_val = indices_val[next_index]

        # janky fix -- need to ensure first element does not have < > and
        # is not a rest
        if (running_length < 0.00001):
            terms = next_val.split(',')
            tries = 0
            while (terms[0] == 'R' or len(terms) != 2):
                # give up after 1000 tries; choose in a guided, random fashion
                if tries >= 1000:
                    print('Gave up on first note generation')
                    rand = np.random.randint(0, len(abstractGrammars))
                    next_val = abstractGrammars[rand].split(' ')[0]

                else:
                    preds = model.predict(x, verbose=0)[0]
                    next_index = sample(preds, diversity)
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

    print(m1_grammar)

    # Pruning #1: 'Smooth' the measure, or make sure that everything is in 
    # standard note lengths (0.125, 0.250, 0.333 ... nothing like .482).
    # Maybe just start with rounding to nearest multiple of 0.125.
    m1_grammar = m1_grammar.split(' ')
    for ix, gram in enumerate(m1_grammar):
        terms = gram.split(',')
        # terms[1] = str(roundDown(float(terms[1]), 0.250))
        terms[1] = str(roundUpDown(float(terms[1]), 0.250, random.choice([-1, 1])))
        m1_grammar[ix] = ','.join(terms)
    m1_grammar = ' '.join(m1_grammar)   

    # get notes from grammar and chords
    m1_notes = unparseGrammar(m1_grammar, m1_chords)

    # fix note offset problems, i.e. same offset so pruning too many.
    # remember - later you can remove 'if (n2.offset - n1.offset) < 0.125' since
    # already adjusted the note durations to be regular enough.
    # QA TODO: chop off notes with offset > 4.0.

    # Another possible fix: get rid of 0.125 length notes ONLY if there are less
    # than three of them in a row.

    # Pruning #2: remove repeated notes, and notes WAY too close together.
    for n1, n2 in grouper(m1_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        # if (n2.offset - n1.offset) < 0.125:
        #     if random.choice(([True] * 10 + [False] * (loopIndex)**2)):
        #         m1_notes.remove(n2)
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                m1_notes.remove(n2)

    # pdb.set_trace()

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
    print('\n')

    # after quality assurance
    # pdb.set_trace()
    for m in m1_notes:
        genStream.insert(currOffset + m.offset, m)
    for mc in m1_chords:
        genStream.insert(currOffset + mc.offset, mc)

    # pdb.set_trace()
    currOffset += 4.0

# genStream.insert(0.0, instrument.ElectricGuitar())
genStream.insert(0.0, tempo.MetronomeMark(number=130))

# Play the final stream (improvisation + accompaniment) through output
# play is defined as a lambda function above
play(genStream)