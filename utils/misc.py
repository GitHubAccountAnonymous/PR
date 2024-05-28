import random


def find_match(l1, l2):
    """
    Returns pair of indices of first match between elements in l1 and l2
    :param l1: First list.
    :param l2: Second list.
    :return: Tuple of indices if match has been found, None otherwise.
    """
    for x in range(len(l1)):
        for y in range(len(l2)):
            if l1[x] == l2[y]:
                return (x, y)
    return None

def sample(seq, n):
    if n <= len(seq):
        return random.sample(seq, n)
    else:
        additional = seq * (n // len(seq))
        additional.extend(random.sample(seq, n % len(seq)))
        return additional