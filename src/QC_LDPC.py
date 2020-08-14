import numpy as np

from utils import gaussjordan

from LinearCode import LinearCode


class QC_LDPC(LinearCode):
    """
    Quasi-Cyclic LDPC code representation (extends LinearCode)

    ...

    Methods
    -------
    from_params(n, p, w)
        Init QC-LDPC by length, circulant size and code weight

    _get_circulant_block(polynom)
        Get circulant (p, p) for given vector of size p
        
    """

    def __init__(self, G, H):
        super().__init__(G, H)
        
    @classmethod
    def from_params(cls, n, p, w):
        assert n % p == 0, "p must be delimeter of n"
        
        n0 = n // p
        assert w > n0, "not enough code weight"
        
        fine = False
        
        while not fine:
            blocks = []
            inverse_block = None
            inverse_block_position = None
            
            vector = [1 for _ in range(w)] + [0 for _  in range(n - w)]
            vector = np.array(vector, dtype=int)
            np.random.shuffle(vector)
            
            for i in range(n0):
                circ = vector[i*p:(i+1)*p]
                
                if sum(circ) < 1:
                    inverse_block = None
                    break
        
                block = QC_LDPC._get_circulant_block(circ)
                blocks.append(block)
                    
                A, P = gaussjordan(block, True)
                A = np.array(A, dtype=int)
                P = np.array(P, dtype=int)
                
                if (A == np.eye(p, dtype=int)).all():
                    inverse_block_position = i
                    inverse_block = P
                
            # continue only if inverse circulant found
            fine = True if inverse_block is not None else False
        
        # put inverse block on last position
        blocks[inverse_block_position], blocks[n0-1] = blocks[n0-1], blocks[inverse_block_position]
        
        for i in range(n0):
            blocks[i] = blocks[i] @ inverse_block % 2
        
        H = np.concatenate(blocks, axis=1)

        for i in range(n0):
            blocks[i] = blocks[i].T

        Ht = np.concatenate(blocks[:n0-1], axis=0)
        G = np.concatenate((np.eye(Ht.shape[0], dtype=int), Ht), axis=1)
        
        assert (G @ H.T % 2 == 0).all(), "G is not correspond to H"
        
        return cls(G, H)
    
    @staticmethod
    def _get_circulant_block(polynom):
        N = len(polynom)
        block = np.empty((N, N), dtype=int)
        
        for i in range(N):
            block[i] = np.roll(polynom, i)
            
        return block


"""
# EXAMPLE USAGE:

n = 200
p = 100
w = 8

qc_ldpc = QC_LDPC.from_params(n, p, w)

word = np.random.randint(2, size=p)
encoded = qc_ldpc.encode(word)

# error vector size n with t or less errors
errors_num = 8
e = [1 for _ in range(errors_num)] + [0 for _  in range(n - errors_num)]
e = np.array(e)
np.random.shuffle(e)

corrupted = (encoded + e) % 2
decoded = qc_ldpc.decode(np.copy(encoded))
decoded = qc_ldpc.get_message(decoded)

assert (decoded == word).all()
"""
