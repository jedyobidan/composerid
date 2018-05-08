import mido
import numpy as np
from os import listdir
import os
import sys
from random import shuffle
import pickle

QNLS_PER_PHRASE = 8
TOKENS_PER_QNL = 4
SAMPLES_PER_PIECE = 100
NOTE_RANGE = 139
TRAIN_SAMPLES = -12500
VAL_SAMPLES = -1000
TEST_SAMPLES = -1000
NCOMPOSERS = 50


KEY_SHIFT = {
    'C': 0, 'B#': 0, 'Cb': 1, 'B': 1, 'Bb': 2, 'A#': 2, 'A': 3, 'Ab': 4, 'G#': 4,
    'G': 5, 'Gb': 6, 'F#': 6, 'F': 7, 'E#': 7, 'Fb': 8, 'E': 8, 'Eb': 9, 'D#': 9,
    'D': 10, 'Db': 11, 'C#': 11, 'unknown': 0
}

class ObjectIndex(object):
    def __init__(self, maximum=-1):
        self.objs = []
        self.max = maximum

    def getIndex(self, obj):
        if obj not in self.objs:
            if len(self.objs) == self.max:
                raise IndexError('maximum value reached')
            self.objs.append(obj)

        return self.objs.index(obj)

    def getObject(self, i):
        return self.objs[i]

    def copyFrom(self, other):
        self.objs = other.objs
        self.max = other.max


Composers = ObjectIndex(NCOMPOSERS)
TimeSignatures = ObjectIndex(50)
KeySignatures = ObjectIndex(52)

NFEATURES = NOTE_RANGE + TimeSignatures.max + KeySignatures.max + 1

class IntervalSet(object):
    def __init__(self):
        self.intervals = []
        self.pos_edge = {}

    def addInterval(self, start, end, value):
        if start < end:
            self.intervals.append((start, end, value))

    def getValuesAt(self, t):
        return [v for (s, e, v) in self.intervals if s <= t and e > t]

    def addPosEdge(self, id, start, value):
        self.pos_edge[id] = (start, value)

    def addNegEdge(self, id, end):
        if id in self.pos_edge:
            edge = self.pos_edge[id]
            self.addInterval(edge[0], end, edge[1])

    def finalize(self, end):
        for id in self.pos_edge:
            self.addNegEdge(id, end)



class MusicPiece(object):
    def __init__(self, composer, path):
        self.path = path
        self.notes = IntervalSet()
        self.tempos = IntervalSet()
        self.time_sigs = IntervalSet()
        self.keys = IntervalSet()
        self.composer = Composers.getIndex(composer)
        self.qnls = 0

        # Default values for metadata
        self.tempos.addPosEdge('tp', 0, 120)
        self.time_sigs.addPosEdge('ts', 0, TimeSignatures.getIndex((4, 4)))
        self.keys.addPosEdge('key', 0, KeySignatures.getIndex('unknown'))

        mid = mido.MidiFile(path)
        for track in mid.tracks:
            time = 0 # In qnls
            for msg in track:
                time += float(msg.time) / mid.ticks_per_beat
                if msg.type == 'note_on':
                    self.notes.addPosEdge(msg.note, time, msg.note)
                elif msg.type == 'note_off':
                    self.notes.addNegEdge(msg.note, time)
                elif msg.type == 'set_tempo':
                    self.tempos.addNegEdge('tp', time)
                    self.tempos.addPosEdge('tp', time, mido.tempo2bpm(msg.tempo))
                elif msg.type == 'time_signature':
                    self.time_sigs.addNegEdge('ts', time)
                    self.time_sigs.addPosEdge('ts', time, 
                        TimeSignatures.getIndex((msg.numerator, msg.denominator)))
                elif msg.type == 'key_signature':
                    self.keys.addNegEdge('key', time)
                    self.keys.addPosEdge('key', time, KeySignatures.getIndex(msg.key))

            self.qnls = max(self.qnls, time)

        self.notes.finalize(self.qnls)
        self.tempos.finalize(self.qnls)
        self.time_sigs.finalize(self.qnls)
        self.keys.finalize(self.qnls)

        self.normalize_notes()

        self.examples = self.getTrainingExamples()

        print 'Processed %s (%d notes, %d keys, %d times, %d tempos, %d examples)' % (
            self.path, 
            len(self.notes.intervals),
            len(self.keys.intervals),
            len(self.time_sigs.intervals),
            len(self.tempos.intervals),
            len(self.examples)
        )

    def fillMat(self, mat, intervals, one_hot=True, offset=0, gran=TOKENS_PER_QNL):
        for (start, end, v) in intervals:
            length = int((end - start) * gran)
            start = int(start * gran)
            if one_hot:
                mat[start:start+length, v + offset] = np.ones(length)
            else:
                mat[start:start+length, offset] = v * np.ones(length)

    def toMat(self, gran=TOKENS_PER_QNL):
        mat = np.zeros((int(self.qnls * gran), NFEATURES))
        self.fillMat(mat, self.notes.intervals, gran=gran)
        self.fillMat(mat, self.tempos.intervals, one_hot=False, offset=NOTE_RANGE, gran=gran)
        self.fillMat(mat, self.time_sigs.intervals, offset=NOTE_RANGE+1, gran=gran)
        self.fillMat(mat, self.keys.intervals, offset=NOTE_RANGE+1+TimeSignatures.max, gran=gran)
        return mat

    def getTrainingExamples(self, qnls=QNLS_PER_PHRASE, gran=TOKENS_PER_QNL, limit=SAMPLES_PER_PIECE):
        mat = self.toMat(gran)
        samples = np.split(mat, np.arange(0, self.qnls*gran, qnls*gran, dtype=np.int32))
        samples = samples[1:-1]
        shuffle(samples)
        if len(samples) > SAMPLES_PER_PIECE:
            samples = samples[:SAMPLES_PER_PIECE]
        return samples

        return mat

    def length(self):
        return self.qnls

    def labelVec(self):
        composer_vec = np.zeros(Composers.max)
        composer_vec[self.composer] = 1
        return composer_vec

    def normalize_notes(self):
        for i, (start, end, note) in enumerate(self.notes.intervals):
            note = self.transpose(note, self.keys.getValuesAt(start)[0])
            self.notes.intervals[i] = (start, end, note)

    def transpose(self, note, key):
        key = KeySignatures.getObject(key)
        if key[-1] == 'm':
            key = key[:-1]

        return note + KEY_SHIFT[key]


def process_dataset(path, split):
    return {c: get_examples('/'.join((path, c, split)), c) for c in listdir(path)}

def get_examples(path, composer):
    examples = []
    for midi in listdir(path):
        try:
            m = MusicPiece(composer, '/'.join((path, midi)))
        except Exception as e:
            print 'Failed to process %s' % '/'.join((path, midi))
            continue
        if m.qnls <= QNLS_PER_PHRASE: # Too short
            continue
        examples.append(m)

    return examples

def save_preprocess_vars(path):
    path = '/'.join((path, 'indices.pkl'))
    pickle.dump((Composers, TimeSignatures, KeySignatures), open(path, 'wb'))

def load_preprocess_vars(path):
    path = '/'.join((path, 'indices.pkl'))
    c, t, k = pickle.load(open(path, 'rb'))
    Composers.copyFrom(c)
    TimeSignatures.copyFrom(t)
    KeySignatures.copyFrom(k)

