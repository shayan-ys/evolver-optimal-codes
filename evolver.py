import helper_methods as hm
import numpy as np
import random
from datetime import datetime

hm.PRINT_BENCHMARKS = True


def pairwise(iterable):
    """
    Iterate over pairs (even-odd)
    :param iterable: s = s0, s1, s2, s3, s4, s5, ...
    :return: (s0, s1), (s2, s3), (s4, s5), ...
    """
    a = iter(iterable)
    return zip(a, a)


def crossover_1_point(parent1: list, parent2: list) -> (list, list):
    parent1_left = parent1[:int(len(parent1) / 2)]
    parent1_right = parent1[int(len(parent1) / 2):]
    parent2_left = parent2[:int(len(parent2) / 2)]
    parent2_right = parent2[int(len(parent2) / 2):]

    child1 = parent1_left + parent2_right
    child2 = parent2_left + parent1_right
    return child1, child2


def generate_probabilities_based_on_len(members) -> list:
    probabilities = [0.0]
    sum_len = 0
    for code in members:
        sum_len += len(code)

    sum_p = 0
    for code in members[1:]:
        p = len(code) / sum_len
        probabilities.append(p)
        sum_p += p
    probabilities[0] = 1 - sum_p
    # Insuring the sum would be 1

    probabilities += [0.0] * (len(members) - len(probabilities))

    return probabilities


def code_validator(codewords_list: list, distance_table: list) -> list:
    validated_code = []
    codewords_shuffled = list(codewords_list)
    random.shuffle(codewords_shuffled)
    for word in codewords_shuffled:
        if hm.is_word_satisfy_minimum_distance_of_code(validated_code, distance_table[word]):
            validated_code.append(word)

    return sorted(validated_code)


def remove_duplicate_codes(population: list) -> list:
    unique_pop = []
    for base_index, base in enumerate(population):
        is_base_unique = True
        for comparator_index, comparator in enumerate(population[base_index + 1:]):
            if len(base) == len(comparator) and base == comparator:
                is_base_unique = False
                break

        if is_base_unique:
            unique_pop.append(base)

    return unique_pop


def selection_tournament(population: list, size_needed: int, probabilities: list, k: int=2) -> list:
    selection = []
    pop_indices = list(range(len(population)))
    for i in range(size_needed):
        selected_chunk_indices = np.random.choice(pop_indices, k, replace=False, p=probabilities)
        selected_chunk = [population[i] for i in selected_chunk_indices]
        selected_best_member = sorted(selected_chunk, key=len)[0]
        selection.append(selected_best_member)

    return selection


class Generation:
    """
    Each member: a code (list of codewords) with length n and minimum distance of d
    Fitness: length of each member (=M)
    Top members of generation: generations are sorted decreasing by their length (fitness) -> first members: fittest
    Mutation: add a random vector to the code
    Crossover: combine first half of a code to second half of another
    Validation: 2-loop over codewords, each codeword has distance of less than d -> out.
    """
    generation_index = 0

    pop_size = 300
    pop_size_upper_limit = pop_size * 1.15

    members = []
    probabilities = None    # based on fitness of members

    selection_percentage = 0.8
    permutation_percentage = 0.3
    permutation_insertion_count = 30
    pride_group_size = 3
    birth_permutation = True

    vectors_universe = None     # cache
    distance_table = []     # cache

    def __init__(self, members: list, distance_table: list=None):
        self.members = members
        if not distance_table:
            vectors_list = hm.lexi_sorter(hm.generate_all_vectors(n))
            self.distance_table = hm.generate_hamming_distance_table(vectors_list, d, print_result=False)
        else:
            self.distance_table = distance_table

        self.vectors_universe = list(range(len(self.distance_table)))

    def selection(self) -> list:
        tournament_selection_chunk_size = 2     # in tournament selection: k
        selected_parents = selection_tournament(self.members, size_needed=int(self.pop_size * self.selection_percentage),
                                                probabilities=self.probabilities, k=tournament_selection_chunk_size)
        return selected_parents

    def permute_member(self, member: list, permuting_genes_count: int) -> None:
        inserting_genes = list(np.random.choice(
            self.vectors_universe, size=permuting_genes_count, replace=False))
        member += inserting_genes

    def permutation(self, population: list, validate_population: bool=False) -> None:
        members_indices = list(range(len(population)))
        to_be_permuted_members_count = min(len(population), int(self.permutation_percentage * self.pop_size))
        to_be_permuted_members = np.random.choice(members_indices, size=to_be_permuted_members_count, replace=False)

        for selected_member_index in to_be_permuted_members:

            self.permute_member(population[selected_member_index], self.permutation_insertion_count)
            if validate_population:
                population[selected_member_index] = code_validator(population[selected_member_index],
                                                                   self.distance_table)

    def crossover(self, selected_parents: list) -> list:
        new_gen = []

        for parent1, parent2 in pairwise(selected_parents):
            child1, child2 = crossover_1_point(parent1, parent2)

            # permutation on new-born children to try to increase the length (create fitter children)
            max_len = len(self.members[0]) + self.permutation_insertion_count
            self.permute_member(child1, permuting_genes_count=max_len - len(child1))
            self.permute_member(child2, permuting_genes_count=max_len - len(child2))

            new_gen += [child1, child2]

        return new_gen

    def new_generation(self, dry_run: bool=False):
        # init for new_gen
        self.probabilities = generate_probabilities_based_on_len(self.members)

        # introducing new members
        selected_parents = self.selection()
        new_gen = self.crossover(selected_parents)
        self.permutation(new_gen)

        # validating
        for i in range(len(new_gen)):
            new_gen[i] = code_validator(new_gen[i], self.distance_table)

        if not dry_run:
            new_gen = self.members[:self.pride_group_size] + new_gen

            new_gen = remove_duplicate_codes(new_gen)
            if len(new_gen) > self.pop_size_upper_limit:
                new_gen = new_gen[:self.pop_size]

            self.members = list(sorted(new_gen, key=len, reverse=True))
            self.generation_index += 1

        return new_gen


def init_generation(size: int, distance_table: list) -> list:
    vectors_len = len(distance_table)
    gen_members = []
    for members_count in range(size):

        first_codeword = random.randint(0, vectors_len-1)
        code = [first_codeword]     # an initial member
        for word in range(first_codeword, vectors_len):

            if hm.is_word_satisfy_minimum_distance_of_code(code, distance_table[word]):
                code.append(word)

        gen_members.append(code)

    return gen_members


n = 8
d = 4
population_size = 5
timer = datetime.now()

vectors = hm.lexi_sorter(hm.generate_all_vectors(n))
# vectors_reverse_map = {vector: index for index, vector in enumerate(vectors)}

print('vectors: ' + str([str(i) + ': ' + ''.join(map(str, vec)) for i, vec in enumerate(vectors)]))

hamming_distance_table = hm.generate_hamming_distance_table(vectors, d, print_result=False)

init_gen = init_generation(population_size, hamming_distance_table)
init_gen = list(sorted(init_gen, key=len, reverse=True))

# for member in init_gen:
#     print([''.join(map(str, vectors[i])) for i in member])

gen = Generation(members=init_gen, distance_table=hamming_distance_table)
print('initial M = ' + str(len(gen.members[0])))

break_M = 16

max_M = 0
for i in range(15000):
    new_pop = gen.new_generation()
    M = len(gen.members[0])
    if M > max_M:
        print('current M = ' + str(M))
        max_M = M
        print('--- computation time so far: ' + str(datetime.now() - timer))
        if M == break_M:
            break

print('---------- computation time: ' + str(datetime.now() - timer))
# print("------ new gen -----")

# for member in gen.members:
#     print([''.join(map(str, vectors[i])) for i in member])
