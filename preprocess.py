'''
Author:     Ji-Sung Kim
Project:    jazzml
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import OrderedDict
from itertools import groupby, izip_longest
from grammar import *

''' Get relevant data from a MIDI file. '''
def get_data(fn):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(fn)

    # Get melody part, compress into single voice.
    # For Metheny piece, Melody is Part #5.
    melodyStream = midi_data[5]
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
    compStream.append([j.flat for i, j in enumerate(midi_data) if i in partIndices])

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

    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melodyStream = soloStream[-1]
    allMeasures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melodyStream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        allMeasures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = soloStream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    allMeasures_chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        allMeasures_chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Fix for the below problem.
    #   1) Find out why len(allMeasures) != len(allMeasures_chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del allMeasures_chords[len(allMeasures_chords) - 1]
    assert len(allMeasures_chords) == len(allMeasures)

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

    corpus = [x for sublist in abstractGrammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))


    return corpus, values, val_indices, indices_val, allMeasures_chords, \
        abstractGrammars