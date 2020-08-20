import numpy as np
import warnings

from random import choice


# originally taken from https://github.com/hichamjanati/pyldpc
def gaussjordan(X, change=0):
    """
    Compute the binary row reduced echelon form of X.

    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform

    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.

    """

    A = np.copy(X)
    m, n = A.shape

    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1

    # for each column
    for j in range(n):
        # take column
        filtre_down = A[pivot_old+1:m, j]
        # find first 1 index in that column
        pivot = np.argmax(filtre_down)+pivot_old+1

        # if it is really 1 on that index
        if A[pivot, j]:
            pivot_old += 1
            # if it is not previous index
            if pivot_old != pivot:
                # swap rows
                aux = np.copy(A[pivot, :])
                A[pivot, :] = A[pivot_old, :]
                A[pivot_old, :] = aux
                if change:
                    aux = np.copy(P[pivot, :])
                    P[pivot, :] = P[pivot_old, :]
                    P[pivot_old, :] = aux

            # for each row
            for i in range(m):
                # if found 1 in intersection
                if i != pivot_old and A[i, j]:
                    if change:
                        P[i, :] = abs(P[i, :]-P[pivot_old, :])
                    # put binary diff in a row
                    A[i, :] = abs(A[i, :]-A[pivot_old, :])

        if pivot_old == m-1:
            break

    if change:
        return A, P
    return A


def bit_flipping(H, c, maxiter=200):
    """
    Bit-flipping algorithm implementation

    Parameters
    ----------
    H: numpy.ndarray. 
        Parity-check matrix
    c : numpy.ndarray
        Codeword
    maxiter : int
        Maximum iterations to perform (default is 2)

    Returns
    -------
    c : numpy.ndarray
        Decoded codeword (raise warning if decoding is incorrect)

    """

    n, k = H.shape
    
    Cn = {i : set() for i in range(n)} # check nodes
    Vn = {i : set() for i in range(k)} # variable nodes
    
    for i in range(n):
        for j in range(k):
            if H[i][j] == 1:
                Cn[i].add(j)
                Vn[j].add(i)
                
    def fix_errors(c, iteration):
        En = np.zeros(k, dtype=int) # total error counter
        Sn = np.zeros(n, dtype=int) # syndrom
        
        # max depth exceeded -> exit with warning
        if iteration == 0:
            message = "Bit flipping failed with {} iterations.".format(maxiter)
            message += " You may need to increase this number."
            warnings.warn(message)
            return
        
        # calculate syndrome
        for node, relations in Cn.items():
            Sn[node] = sum(c[list(relations)]) % 2
        
        # count errors
        for node, check in enumerate(Sn):
            # if check is not satisfied
            if check == 1:
                for relation in Cn[node]:
                    En[relation] += 1
                    
        # if syndrom is correct -> exit
        if sum(En) == 0:
            return
        
        # find indexes with max error
        m = 0
        max_indexes = []
        for i, elem in enumerate(En):
            if elem == m and m > 0:
                max_indexes.append(i)
            elif elem > m:
                max_indexes = []
                m = elem
                max_indexes.append(i)
            
        # flip one random element of them
        flip = choice(max_indexes)
        c[flip] = 1 if c[flip] == 0 else 0
        
        # go down recursively
        fix_errors(c, iteration-1)
    
    fix_errors(c, maxiter)
    
    return c
