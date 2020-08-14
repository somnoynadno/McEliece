import numpy as np

from random import choice
from pyldpc import make_ldpc

from LinearCode import LinearCode


class LDPC(LinearCode):
	"""
	Low-Density Parity-Check code (extends LinearCode)

	...
	Methods
	-------
	from_params(n, d_v, d_c)
		Init LDPC by size of H, column weight and row weight

	"""

	def __init__(self, G: np.ndarray, H: np.ndarray):       
		super().__init__(G, H)

	@classmethod
	def from_params(cls, n, d_v, d_c):
		H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

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
