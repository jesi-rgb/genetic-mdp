import numpy as np
from itertools import combinations, permutations, product, groupby
import multiprocessing as mp

from pprint import pprint


    



def create_random_solution(n):
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # creamos valores aleatorios para los índices elegidos
    chosen = np.random.rand(n)

    # aquellos índices que superen un umbral se ponen a 1
    M[chosen > 0.5] = 1

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


def brute_force(n, D, max_m):
    # extract all posible combinations of max_m cardinality to
    # reduce search space
    all_combs = [np.array(i) 
                for i in product([0, 1], repeat=n)]

    filtered_combs = list(filter(lambda x: np.sum(x)==max_m, all_combs))

    print("Espacio de posibles soluciones para {} elementos y m = {}: {}\n".format(n, max_m, len(filtered_combs)))

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



def local_search(n, D, max_m):
    M = np.zeros(n, dtype=int)
    M[n-max_m:n] = 1
    M = np.random.permutation(M)

    curr_div = calculate_diversity(M, D)

    # calculate neighbourhood
    neighbours(M)



if __name__ == "__main__":
    np.random.seed(9)

    n = 20 # número de elementos en nuestro array original
    max_m = 8

    # creamos una matriz de distancias para cada pareja de elementos
    # la diagonal principal será 0 porque d(i, i) = 0
    D = np.random.randint(100, size=(n, n))
    np.fill_diagonal(D, 0)

    # creamos una solución aleatoria

    print("La matriz de distancias es:\n")
    print(D)
    print()
    print()
    brute_force(n, D, max_m)
    # local_search(n, D, max_m)
    # calculate_sizes(n, max_m)