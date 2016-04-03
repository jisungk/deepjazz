'''
Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.

'''

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
    for i in notes: print pretty(i)

''' Print list and stuff. '''
def compareGen(m1_grammar, m1_elements):
    for i, j in zip(m1_grammar.split(' '), m1_elements):
        if isinstance(j, note.Note):
            print '%s  |  %s' % (ppn(j), i)
        else:
            print '%s  |  %s' % (ppr(j), i)

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
# from grammar import parseMelody, unparseGrammar
from grammar import *
# from findSolo import findSolo

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

# FOR REFERENCE
# fullMelody.append(key.KeySignature(sharps=1, mode='major'))

# Bad notes: melodyVoice[370, 391]
# the two B-4, B-3 around melodyVoice[362]. I think: #362, #379
# REMEMBER: after delete each of these, 
# badIndices = [370, 391, 362, 379]
# BETTER FIX: iterate through, if length = 0, then bump it up to 0.125
# since I think this is giving the bug.
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
partIndices = [0, 1, 6, 7] # TODO: remove 0.0==inf duration bug in part 7
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

''' Could make alternative version where everything lined up with beats 1 and 3.
    
    Iterate over all measures, generating the grammar strings for each measure. You can do this in non-realtime - because you're just 
    generating the list of clusters.

    Note: since you're just generating cluster patterns, abstractGrammars
    does not strictly HAVE to be the same length as the # of measures, 
    although it would be nice. 
'''
# TRAINING DATA
abstractGrammars = []
# pdb.set_trace()
for ix in xrange(1, len(allMeasures)):
    # pdb.set_trace()
    m = stream.Voice()
    for i in allMeasures[ix]:
        m.insert(i.offset, i)
    c = stream.Voice()
    for j in allMeasures_chords[ix]:
        c.insert(j.offset, j)
    parsed = parseMelody(m, c)
    # print '%s: %s\n' % (ix, parsed)
    abstractGrammars.append(parsed)

genStream = stream.Stream()

# Where to start, end loop
currOffset = 0
loopEnd = len(allMeasures_chords)

#---------------------------------PREPROCESS2----------------------------------#
corpus = [x for sublist in abstractGrammars for x in sublist.split(' ')]












#----------------------------------GENERATE------------------------------------#
'''
Code adapted from public examples in the Keras documentation: 
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py


At least 20 epochs are required before the generated text
starts sounding coherent.
'''

'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

for loopIndex in range(1, loopEnd): # prev: len(allMeasures_chords)
    # get chords from input
    m1_chords = stream.Voice() # initialize
    for j in allMeasures_chords[loopIndex]:
        m1_chords.insert((j.offset % 4), j)



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

    # get ntoes from grammar and chords
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
    print 'After pruning: %s notes' % (len([i for i in m1_notes
        if isinstance(i, note.Note)]))
    print '\n'

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
# play(genStream)
'''