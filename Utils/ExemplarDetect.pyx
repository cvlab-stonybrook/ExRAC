# max_product_cython.pyx
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple max_product_cy(list A):
    cdef int n = len(A)
    cdef double max_A1i = A[0][0]
    cdef int max_A1i_idx = 0
    cdef double max_A1i_A2j = 0
    cdef tuple max_A1i_A2j_idx = (0, 0)
    cdef double max_product = 0

    cdef int best_i = 0
    cdef int best_j = 0
    cdef int best_k = 0

    cdef double a1, a2, a3
    cdef double current_product

    for k in range(1, n):
        a1, a2, a3 = A[k]
        current_product = max_A1i_A2j * a3

        if current_product > max_product:
            max_product = current_product
            best_i, best_j = max_A1i_A2j_idx
            best_k = k

        if a2 != 0 and max_A1i * a2 > max_A1i_A2j:
            max_A1i_A2j = max_A1i * a2
            max_A1i_A2j_idx = (max_A1i_idx, k)

        if a1 > max_A1i:
            max_A1i = a1
            max_A1i_idx = k

    return max_product, [best_i, best_j, best_k]