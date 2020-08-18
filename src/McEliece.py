import numpy as np

from random import randint

from LinearCode import LinearCode
from utils import gaussjordan


class McEliece:
    """
    McEliece cryptosystem representation

    For more info: https://en.wikipedia.org/wiki/McEliece_cryptosystem

    ...

    Attributes
    ----------
    S : numpy.ndarrray
        random matrix (k, k)
    P : numpy.ndarrray
        random permutation matrix (n, n)
    t : int
        maximum code error
    code : LinearCode
        linear code using in decryption
    public_key : tuple
        (SGP, t)
    private key : tuple
        (S, G, P)

    Methods
    -------
    from_linear_code(code, t)
        Init cryptosystem with LinearCode with max error t

    encrypt(word)
        Encryption with public key

    decrypt(codeword)
        Decryption with private key

    _get_non_singular_matrix(k)
        Little helper to get inversable matrix (size k)
        
    """

    def __init__(self, code, S, P, t):
        self.S = S
        self.P = P
        self.t = t

        self.code = code

        # sizes
        self.k, self.n = code.getG().shape
                
        # McEliece keys
        self.public_key = ((self.S @ code.getG() @ self.P % 2), self.t)
        self.private_key = (self.S, code.getG(), self.P)
                    
    @classmethod 
    def from_linear_code(cls, code: LinearCode, t: int):
        k, n = code.getG().shape
        
        # permutation matrix (n * n)
        P = np.eye(n, dtype=int) 
        np.random.shuffle(P) 
        
        # random matrix (k * k)
        S = McEliece._get_non_singular_random_matrix(k)
        
        return cls(code, S, P, t)
        
    def encrypt(self, word):
        errors_num = self.t
        
        # error vector size n with t errors
        z = [1 for _ in range(errors_num)] + [0 for _  in range(self.n - errors_num)]
        z = np.array(z, dtype=int)
        np.random.shuffle(z)

        res = ((word @ self.public_key[0] % 2) + z) % 2
            
        return res
    
    def decrypt(self, codeword):
        A, invP = gaussjordan(self.P, True)

        c = codeword @ invP % 2
        c = np.array(c, dtype=int)
        
        d = self.code.decode(c)
        m = self.code.get_message(d)

        _, invS = gaussjordan(self.S, True)
        
        res = m @ invS % 2
        res = np.array(res, dtype=int)
        
        return res
    
    @staticmethod
    def _get_non_singular_random_matrix(k):
        while True:
            S = np.random.randint(0, 2, (k, k)) 

            A = gaussjordan(S)
            A = np.array(A, dtype=int)

            if (A == np.eye(k, dtype=int)).all():
                return S

"""
# EXAMPLE USAGE

from LDPC import LDPC

n = 300
d_v = 6
d_c = 10

ldpc = LDPC.from_params(n, d_v, d_c)

word = np.random.randint(2, size=ldpc.getG().shape[0])
print("word:", word)

crypto = McEliece.from_linear_code(ldpc, 12)

encrypted = crypto.encrypt(word)
print("encrypted:", encrypted)

decrypted = crypto.decrypt(encrypted)
print("decrypted:", decrypted)

assert (word == decrypted).all()
"""
