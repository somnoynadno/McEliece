import numpy as np
import warnings

from random import choice
from pyldpc import coding_matrix_systematic, make_ldpc

from LinearCode import LinearCode

from utils import gaussjordan


class LDPC(LinearCode):
	"""
	Low-Density Parity-Check code (extends LinearCode)

	...
	Methods
	-------
	from_params(n, d_v, d_c, regular=True)
		Init LDPC by size of H, column weight and row weight

	"""

	def __init__(self, G: np.ndarray, H: np.ndarray):       
		super().__init__(G, H)

	@classmethod
	def from_params(cls, n, d_v, d_c, regular=True):
		if regular:
			# Gallagher's algorithm
			H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
		else:
			# MacKay's algorithm
			p = n // d_c

			vector = [1 for _ in range(d_c)] + [0 for _ in range(n - d_c)]
			vector = np.array(vector, dtype=int)

			matrix = np.array([np.random.permutation(vector) for _ in range(d_v * p)], dtype=int)
			H, G = coding_matrix_systematic(matrix)

		return cls(G.T, H)		


"""
# EXAMPLE USAGE

n = 300
d_v = 6
d_c = 10

ldpc = LDPC.from_params(n, d_v, d_c)

word = np.random.randint(2, size=ldpc.getG().shape[0])
encoded = ldpc.encode(word)

# error vector size n with t or less errors
t = 12
e = [1 for _ in range(t)] + [0 for _  in range(n - t)]
e = np.array(e)
np.random.shuffle(e)

enc_e = (encoded + e) % 2
print("encoded with {} errors:".format(t), enc_e)

res = ldpc.decode(np.copy(enc_e))
print("decoded:", res)

d = sum(enc_e != res)
print("error in {} bits".format(d))

assert d == t
"""
