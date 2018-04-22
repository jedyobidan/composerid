import mido
import numpy as np

QNLS_PER_PHRASE = 4
TOKENS_PER_QNL = 8
NOTE_RANGE = 128
ORTHO_FEATURES = 3

class ObjectsToTokens(object):
	def __init__(self):
		self.items = []

	def get(self, item):
		if item not in self.items:
			self.items.append(item)

		return self.items.index(item)

COMPOSERS = ObjectsToTokens()
TIME_SIG = ObjectsToTokens()
KEYS = ObjectsToTokens()


class IntervalSet(object):
	def __init__(self):
		self.intervals = []
		self.pos_edge = {}

	def addInterval(self, start, end, value):
		if start < end:
			self.intervals.append((start, end, value))

	def getValuesAt(self, t):
		return [tup[2] for tup in self.intervals if tup[0] <= t and tup[1] > t]

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
		self.tempos = IntervalSet()
		self.time_sigs = IntervalSet()
		self.keys = IntervalSet()
		self.notes = IntervalSet()
		self.composer = COMPOSERS.get(composer)
		self.qnls = 0

		# Default values for metadata
		self.tempos.addPosEdge('tp', 0, 120)
		self.time_sigs.addPosEdge('ts', 0, TIME_SIG.get((4, 4)))
		self.keys.addPosEdge('key', 0, KEYS.get('unknown'))

		mid = mido.MidiFile(path)
		for track in mid.tracks:
			time = 0 # In qnls
			for msg in track:
				time += float(msg.time) / mid.ticks_per_beat
				if msg.type == 'time_signature':
					self.time_sigs.addNegEdge('ts', time)
					self.time_sigs.addPosEdge('ts', time, TIME_SIG.get((msg.numerator, msg.denominator)))
				elif msg.type == 'key_signature':
					self.keys.addNegEdge('key', time)
					self.keys.addPosEdge('key', time, KEYS.get(msg.key))
				elif msg.type == 'set_tempo':
					self.tempos.addNegEdge('tp', time)
					self.tempos.addPosEdge('tp', time, mido.tempo2bpm(msg.tempo))
				elif msg.type == 'note_on':
					self.notes.addPosEdge(msg.note, time, msg.note)
				elif msg.type == 'note_off':
					self.notes.addNegEdge(msg.note, time)

			self.qnls = max(self.qnls, time)

		self.tempos.finalize(self.qnls)
		self.time_sigs.finalize(self.qnls)
		self.keys.finalize(self.qnls)
		self.notes.finalize(self.qnls)

	def getTrainingExamples(self, qnls=QNLS_PER_PHRASE, gran=TOKENS_PER_QNL):
		return [self.featureSeq(s, qnls, gran) for s in np.arange(0, self.qnls, qnls) 
				if s + qnls <= self.qnls]

	def featureSeq(self, start, qnls=QNLS_PER_PHRASE, gran=TOKENS_PER_QNL):
		mat = np.zeros((TOKENS_PER_QNL * QNLS_PER_PHRASE, NOTE_RANGE + ORTHO_FEATURES + 1))
		for i, t in enumerate(np.arange(start, start+qnls, float(1)/gran)):
			mat[i] = self.featuresAt(t)

		return mat

	def featuresAt(self, t):
		vec = np.zeros(NOTE_RANGE + ORTHO_FEATURES + 1)
		for note in self.notes.getValuesAt(t):
			vec[note] = 1

		vec[NOTE_RANGE] = self.tempos.getValuesAt(t)[0]
		vec[NOTE_RANGE+1] = self.time_sigs.getValuesAt(t)[0]
		vec[NOTE_RANGE+2] = self.keys.getValuesAt(t)[0]

		vec[-1] = self.composer
		return vec

