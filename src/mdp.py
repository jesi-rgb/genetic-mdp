from itertools import product, groupby
from math import factorial

from pprint import pprint

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def create_random_solution(n, m):
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # aquellos índices que superen un umbral se ponen a 1
    M[np.random.rand(n) > 0.3] = 1

    M = shape_solution(M, m)
    return M


def read_distance_matrix(path):
    with open(path) as file:
        lines = file.readlines()


    print("Reading {} lines".format(len(lines) - 1))
    n, m = lines[0].split(" ")
    n, m = int(n), int(m)
    data = np.array([float(line.strip().split(" ")[2]) for line in lines[1:]])
    return n, m, data


def fill_upper_triangular(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=float)
    out[mask] = a
    return out.T


def shape_solution(M, m):
    M = np.array(M)
    while np.sum(M) > m:
        ones = M.nonzero()[0]
        random_one = np.random.choice(ones)
        M[random_one] = 0

    while np.sum(M) < m:
        zeros = np.where(M==0)[0]
        random_zero = np.random.choice(zeros)
        M[random_zero] = 1

    return M

   
def calculate_diversity(M, D):
    
    '''
    This function calculates the diversity of a solution based on the
    definition for the Maximum Diversity Problem.
    '''

    # first, find all the possible combination of indices that lie
    # within the upper triangular section of our n x n matrix
    indices_triu = np.argwhere(np.triu(np.ones((len(M),)*2),1))
    indices_triu = indices_triu[indices_triu[:,0] < indices_triu[:,1]]

    # second, find all posible combinations of genotypes
    # within our particular solution
    mesh = np.array(np.meshgrid(M, M))
    combs = mesh.T.reshape(-1, 2)

    # third, calculate all the possible combinations of genotypes, 
    # just as before, but now with the indices, not the values themselves
    M_i = np.indices(np.array(M).shape)
    mesh_i = np.array(np.meshgrid(M_i, M_i))
    combs_i = mesh_i.T.reshape(-1, 2)

    # given those, find all combinations that match our
    # upper triangular section rule, just like before
    combs_triu = combs[combs_i[:,0] < combs_i[:,1]]

    # stack the results so we have rows in the format: [gen1, gen2, gen1_index, gen2_index]
    col_stack = np.column_stack((combs_triu, indices_triu))

    # to calculate the diversity, access the distances matrix with
    # gen1_index and gen2_index, and multiply by the values gen1 and gen2 themselves.
    # If some of the values is 0, this will all be canceled out. 
    # This will return a vector that will contain either the value 
    # of the distance between two particular genes, or 0. Sum it all up and return.
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


def mutation(M, m, m_factor):
    return shape_solution([genotype if np.random.rand() > m_factor else 1 - genotype for genotype in M], m)




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


def genetic_algorithm(n, m, D, initial_population, k_top, m_factor, n_iterations, patience):
    # Initialize population
    # For this, lets generate 1 possible solution and calculate permutations over it
    current_generation = [create_random_solution(n, m) for i in range(initial_population)]

    current_best_solution_d = 0
    last_best_solution = 0
    counter = 0
    for i in range(n_iterations):
        print("*** Iteration {} ***\n\n".format(i))
        # Calculate diversity of each one
        diversity_arr = [calculate_diversity(s, D) for s in current_generation]


        # Take the best k ones and crossover
        gen_div = list(zip(current_generation, diversity_arr)) 
        sorted_gen_div = sorted(gen_div, key = lambda x: x[1], reverse=True)


        print("Best solution in gen {} has diversity {}\n".format(i, sorted_gen_div[0][1]))
        best_solutions = [s[0] for s in sorted_gen_div]
        survivals = best_solutions[:k_top]

        pairs = np.squeeze(sliding_window_view(survivals, (2, n)))

        current_generation = [two_point_crossover(pair[0], pair[1], m) for pair in pairs]
        current_generation = np.reshape(current_generation, (2 * (k_top - 1), n))

        if current_best_solution_d < sorted_gen_div[0][1]:
            current_best_solution_d = sorted_gen_div[0][1]
            current_best_solution = sorted_gen_div[0][0]
            counter = 0

        if(last_best_solution == current_best_solution_d):
            counter += 1
        else:
            last_best_solution = current_best_solution_d

        print("Best solution so far has diversity {}\n".format(last_best_solution))
        print("Patience counter: {}. {} more to finish if equal.".format(counter, patience - counter))
        if counter == patience:
            print("Value stabilized at {} with solution {}".format(current_best_solution_d, current_best_solution))
            return (current_best_solution, current_best_solution_d)

    # Mutation
        current_generation = [mutation(solution, m, m_factor) for solution in current_generation]




if __name__ == "__main__":
    # np.random.seed(7)

    n, m, data = read_distance_matrix("src/data/MDG-a_1_n500_m50.txt")
    

    print(data)
    # # creamos una matriz de distancias para cada pareja de elementos
    D = fill_upper_triangular(data)

    # D = np.random.randint(100, size=(n, n), dtype=int)

    # creamos una solución aleatoria

    print("La matriz de distancias es:\n")
    print(D)
    print()
    print()

    genetic_algorithm(n, m, D, initial_population=100, k_top=15, m_factor=0.002, n_iterations=100, patience=20)

    total_space = factorial(n) // (factorial(n - m) * factorial(m))
    print("\nEl espacio de búsqueda consta de {} posibles soluciones para n = {} y m = {}.".format(total_space, n, m))

