import numpy as np
from itertools import combinations, permutations, product, groupby
import multiprocessing as mp

from pprint import pprint

from numpy.random import random
from math import factorial

def create_random_solution(n, m):
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # aquellos índices que superen un umbral se ponen a 1
    M[np.random.rand(n) > 0.3] = 1

    while np.sum(M) > m:
        ones = M.nonzero()[0]
        random_one = np.random.choice(ones)
        M[random_one] = 0

    while np.sum(M) < m:
       zeros = np.nonzero(M==0)[0]
       random_zero = np.random.choice(zeros)
       M[random_zero] = 1

    return M






def calculate_diversity(M, D):
    mesh = np.array(np.meshgrid(M, M))
    combs = mesh.T.reshape(-1, 2)
    
    M_i = np.indices(np.array(M).shape)
    mesh_i = np.array(np.meshgrid(M_i, M_i))
    combs_i = mesh_i.T.reshape(-1, 2)

    col_stack = np.column_stack((combs, combs_i))

    return np.sum([D[ i[2], i[3] ] * i[0] * i[1] 
                    for i in col_stack])


def brute_force(n, D, m):
    # extract all posible combinations of m cardinality to
    # reduce search space
    all_combs = [np.array(i) 
                for i in product([0, 1], repeat=n)]

    filtered_combs = list(filter(lambda x: np.sum(x)==m, all_combs))

    print("Espacio de posibles soluciones para {} elementos y m = {}: {}\n".format(n, m, len(filtered_combs)))

    all_solutions = [calculate_diversity(c, D) for c in filtered_combs]

    index = np.argmax(all_solutions)
    max_v = all_solutions[index]
    best_comb = filtered_combs[index]
    print("La mejor solución es {}, en el índice {} con una diversidad de {}".format(best_comb, index, max_v))

def swap(M, index, left=True):
    # flip
    if left:
        if M[index - 1] != 1:
            M[index], M[index - 1] = M[index - 1], M[index]
        else:
            M[index], M[index + 1] = M[index + 1], M[index]
    else:
        if M[index + 1] != 1:
            M[index], M[index + 1] = M[index + 1], M[index]
        else:
            M[index], M[index - 1] = M[index - 1], M[index]

    return M

def neighbours(M):
    # detect the ones
    indices = np.indices(M.shape).flatten()[M==True]

    meta_indices = np.indices(indices.shape).flatten()

    ole = np.column_stack((meta_indices, indices))

    gb = groupby(ole, key=lambda x: x[0] - x[1])
    all_groups = ([i[1] for i in g] for _, g in gb)

    bad_indices = list(filter(lambda x: len(x) > 1, all_groups))

    list(map(lambda x: [x[0], x[-1]], bad_indices))

    pprint(indices)

    ole = [swap(M.copy(), i) for i in indices]
    pprint(ole)



def local_search(n, D, m):
    M = np.zeros(n, dtype=int)
    M[n-m:n] = 1
    M = np.random.permutation(M)

    curr_div = calculate_diversity(M, D)

    # calculate neighbourhood
    neighbours(M)



def genetic_algorithm(n):
    # Initialize population
    # For this, lets generate 1 possible solution and calculate permutations over it
    first_generation = [create_random_solution(n) for i in range(10)]

    # Calculate diversity of each one
    diversity_arr = [calculate_diversity(s) for s in first_generation]

    # Take the best k ones and crossover
    

    # Mutation




if __name__ == "__main__":
    np.random.seed(1)

    n = 20 # número de elementos en nuestro array original
    m = 10 
    
    total_space = factorial(n) // (factorial(n - m) * factorial(m))
    print("El espacio de búsqueda consta de {} posibles soluciones para n = {} y m = {}.".format(total_space, n, m))

    # creamos una matriz de distancias para cada pareja de elementos
    # la diagonal principal será 0 porque d(i, i) = 0
    D = np.random.randint(100, size=(n, n))
    np.fill_diagonal(D, 0)

    # creamos una solución aleatoria

    print("La matriz de distancias es:\n")
    print(D)
    print()
    print()
    # brute_force(n, D, m)
    # local_search(n, D, m)
    # calculate_sizes(n, m)


