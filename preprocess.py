import mido
import numpy as np

QNLS_PER_PHRASE = 4
TOKENS_PER_QNL = 8
NOTE_RANGE = 128

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


Composers = ObjectIndex(30)
TimeSignatures = ObjectIndex(8)
KeySignatures = ObjectIndex(13)


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

	def getTrainingExamples(self, qnls=QNLS_PER_PHRASE, gran=TOKENS_PER_QNL):
		label = self.labelVec()
		return [(self.featureSeq(s, qnls, gran), label) for s in np.arange(0, self.qnls, qnls) 
				if s + qnls <= self.qnls]

	def featureSeq(self, start, qnls=QNLS_PER_PHRASE, gran=TOKENS_PER_QNL):
		return np.array([self.featuresAt(t) for t in np.arange(start, start+qnls, float(1)/gran)])

	def featuresAt(self, t):
		notes_vec = np.zeros(NOTE_RANGE)
		notes_vec[self.notes.getValuesAt(t)] = 1

		tempo = self.tempos.getValuesAt(t)[0]

		time_sig_vec = np.zeros(TimeSignatures.max)
		time_sig_vec[self.time_sigs.getValuesAt(t)] = 1

		key_vec = np.zeros(KeySignatures.max)
		key_vec[self.keys.getValuesAt(t)] = 1

		return np.r_[notes_vec, tempo, time_sig_vec, key_vec]

	def labelVec(self):
		composer_vec = np.zeros(Composers.max)
		composer_vec[self.composer] = 1
		return composer_vec

