import numpy as np

from utils import bit_flipping


class LinearCode:
	"""
	LiearCode base class

	...

	Attributes
	----------
	G : numpy.ndarray
	    generator matrix
	H : numpy.ndarray
		parity-check matrix

	Methods
	-------
	encode(word)
	    Encode given word by coding matrix

	decode(codeword)
		Decode codeword with bit-flipping algorithm

	get_message(decoded)
		Extract message from decoded word (removing check bits)

	syndrome(codeword)
		Find syndrome for given codeword

	guess_code_max_error(iters=100, confidence=0.999)
		Try to guess maximum error to decode codeword correctly
		
	"""

	def __init__(self, G, H):
		self.G = G
		self.H = H

		# check that H is corresponding to G
		assert (self.G @ self.H.T % 2 == 0).all()

	def encode(self, word):
		return (word @ self.G) % 2

	def decode(self, codeword):
		return bit_flipping(self.H, codeword)

	def get_message(self, decoded):
		return decoded[:self.G.shape[0]] 

	def syndrome(self, codeword):
		return (codeword @ self.H.T) % 2 

	def guess_code_max_error(self, iters=100, confidence=0.999):
		k, n = self.G.shape

		t = 1
		while True:
			counter = 0
			for i in range(iters):
				word = np.random.randint(2, size=k)
				enc = (word @ self.G) % 2

				e = [1 for _ in range(t)] + [0 for _  in range(n - t)]
				e = np.array(e)
				np.random.shuffle(e)

				enc_e = (enc + e) % 2

				res = self.decode(enc_e.copy())
				d = sum(enc_e != res)

				if d == t:
					counter += 1

			if counter / iters < confidence:
				return t - 1
			else:
				t += 1

	def getG(self):
		return self.G

	def getH(self):
		return self.H