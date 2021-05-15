import numpy as np
from itertools import combinations, permutations, product, groupby
import multiprocessing as mp

from pprint import pprint
from numpy.core.fromnumeric import shape

from numpy.lib.stride_tricks import sliding_window_view

from numpy.random import random
from math import factorial

def create_random_solution(n, m):
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # aquellos índices que superen un umbral se ponen a 1
    M[np.random.rand(n) > 0.3] = 1

    M = shape_solution(M, m)
    return M


def shape_solution(M, m):
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
    M[n - m:n] = 1
    M = np.random.permutation(M)

    curr_div = calculate_diversity(M, D)

    # calculate neighbourhood
    neighbours(M)

def classic_crossover(father, mother, m):
    # take random point
    rand_index = np.random.randint(1, len(father) - 1)
    # copy first half of father to child
    lh_f = father[0:rand_index] # left hand father
    rh_f = father[rand_index:] # right hand father
    # copy last half of mother to child
    lh_m = mother[0:rand_index] # left hand mother
    rh_m = mother[rand_index:] # right hand mother
 
    # creation of the new generation
    child_1 = np.concatenate((lh_f, rh_m))
    child_2 = np.concatenate((lh_m, rh_f))

    # make sure the solutions are valid
    child_1 = shape_solution(child_1, m)
    child_2 = shape_solution(child_2, m)
    return (child_1, child_2)




def two_point_crossover(father, mother, m):
    point_1 = np.random.randint(1, len(father) // 2)
    point_2 = np.random.randint(point_1 + 1, len(father) - 1)

    inside_f = father[point_1:point_2]
    inside_m = mother[point_1:point_2]

    outside_l_f = father[0:point_1]
    outside_r_f = father[point_2:]

    outside_l_m = mother[0:point_1]
    outside_r_m = mother[point_2:]

    child_1 = np.concatenate((outside_l_f, inside_m, outside_r_f))
    child_2 = np.concatenate((outside_l_m, inside_f, outside_r_m))
    
    child_1 = shape_solution(child_1, m)
    child_2 = shape_solution(child_2, m)

    return (child_1, child_2)


def genetic_algorithm(n, m, D, initial_population, k_top, n_iterations, patience):
    # Initialize population
    # For this, lets generate 1 possible solution and calculate permutations over it
    current_generation = [create_random_solution(n, m) for i in range(initial_population)]

    current_best_solution = 0
    last_best_solution = 0
    counter = 0
    for i in range(n_iterations):
        print("*** Iteration {} ***\n\n".format(i))
        # Calculate diversity of each one
        diversity_arr = [calculate_diversity(s, D) for s in current_generation]

        # print("Diversity array")
        # print(diversity_arr)

        # Take the best k ones and crossover
        gen_div = list(zip(current_generation, diversity_arr)) 
        sorted_gen_div = sorted(gen_div, key = lambda x: x[1], reverse=True)

        # print("Sorted gen")
        # pprint(sorted_gen_div)

        print("Best solution in gen {} has diversity {}\n".format(i, sorted_gen_div[0][1]))
        best_solutions = [s[0] for s in sorted_gen_div]
        survivals = best_solutions[:k_top]

        pairs = np.squeeze(sliding_window_view(survivals, (2, n)))

        current_generation = [classic_crossover(pair[0], pair[1], m) for pair in pairs]
        current_generation = np.reshape(current_generation, (2 * (k_top - 1), n))

        if current_best_solution < sorted_gen_div[0][1]:
            current_best_solution = sorted_gen_div[0][1]
            counter = 0

        if(last_best_solution == current_best_solution):
            counter += 1
        else:
            last_best_solution = current_best_solution

        print("Best solution so far has diversity {}\n".format(last_best_solution))
        print("Patience counter: {}. {} more to finish if equal.".format(counter, patience - counter))
        if counter == patience:
            print("Value stabilized at {} with solution {}".format(sorted_gen_div[0][1], sorted_gen_div[0][0]))
            break

    # Mutation




if __name__ == "__main__":
    # np.random.seed(7)

    n = 400 # número de elementos en nuestro array original
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

    genetic_algorithm(n, m, D, 10, 5, 100_000, 50)



    # sol_1 = create_random_solution(n, m)
    # sol_2 = create_random_solution(n, m)
    # print("Father: ", sol_1)
    # print("Mother: ", sol_2)
    # c_1, c_2 = two_point_crossover(sol_1, sol_2, m)
    # print("Child 1:", c_1)
    # print("Child 2:", c_2)


