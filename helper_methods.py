import numpy as np
import itertools
from datetime import datetime

PRINT_BENCHMARKS = False


def hamming_distance(vector_1: np.array, vector_2: np.array) -> int:
    return np.count_nonzero(vector_1 != vector_2)


def generate_all_vectors(length: int) -> list:
    """
    Creates all possible binary numbers with length n, and casts them to np.array
    :param length: n
    :return: list of np.arrays representing all possible binary numbers of length n
    """
    return list(map(np.array, itertools.product([0, 1], repeat=length)))


def generate_hamming_distance_table(vector_list: list, minimum_distance: int, print_result: bool=False) -> list:
    """
    Generate a hamming distance table with integer indexes as in the input vectors list, and a Boolean value
    Based on if they satisfy the given minimum distance or not.
    :param vector_list: List of vectors
    :param minimum_distance: Each two vectors that are 'minimum_distance' away from each other will be flagged as 'True'
    :return: Hamming distance of vectors (in order with integer indexes)
    """
    global PRINT_BENCHMARKS
    distance_table_timer = datetime.now()

    distance_table = []

    for needle_index, vector_needle in enumerate(vector_list):

        distance_table.append([])
        for in_stack_index, vector_in_stack in enumerate(vector_list):

            if needle_index == in_stack_index:
                is_distance = False
            elif needle_index > in_stack_index:
                is_distance = distance_table[in_stack_index][needle_index]
            else:
                is_distance = hamming_distance(vector_needle, vector_in_stack) >= minimum_distance

            distance_table[needle_index].append(is_distance)

    if PRINT_BENCHMARKS:
        print('--- distance table pre-computation time: ' + str(datetime.now() - distance_table_timer))

    if print_result:
        for row in distance_table:
            print(row)

    return distance_table


def lexi_sorter(vectors_list: list) -> list:
    return sorted(vectors_list, key=np.count_nonzero)


def is_word_satisfy_minimum_distance_of_code(code: list, hamming_distance_list_for_word: list) -> bool:
    for codeword in reversed(code):
        if not hamming_distance_list_for_word[codeword]:
            return False
    return True