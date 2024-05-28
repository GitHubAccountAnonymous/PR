import re


def get_non_alphanum(text):
    text = re.sub('[a-zA-Z0-9]', '', text)
    text = text.replace(' ', '')
    non_alphanum = set(text)
    return non_alphanum


def get_punctuation(text):
    punct = text.lower()
    unpunct = text.lower()
    marks = get_non_alphanum(unpunct)
    for p in marks:
        if p != "'":
            unpunct = unpunct.replace(p, "")
    unpunct_words = unpunct.split()

    labels = []
    for word in unpunct_words:
        try:
            assert punct.find(word) == 0
        except AssertionError:
            return {'Input': [], 'Label': []}

        idx = len(word)  # index of character immediately after word
        if idx == len(punct):
            labels.append(0)
            break
        else:
            next_word = idx
            while is_special_mark(punct[next_word]):
                next_word += 1

                if next_word == len(punct):
                    next_word = None
                    break

            symbols = list(punct[idx:next_word])
            is_space = [symbol == ' ' for symbol in symbols]

            # There is no punctuation (NP) after this word
            if all(is_space):
                labels.append(0)
            else:
                is_not_space_idx = is_space.index(False)
                symbol = symbols[is_not_space_idx]
                if symbol == '.':
                    labels.append(1)
                elif symbol == ',':
                    labels.append(2)
                elif symbol == '?':
                    labels.append(3)
                else:
                    raise AssertionError("Unrecognized symbol: " + symbol)

            punct = punct[next_word:]

    return labels


def is_special_mark(s):
    if s.isalnum():
        return False
    elif s == "'":
        return False
    else:
        return True


def remove_double_spaces(s):
    """
    Replaces excessive adjacent spaces with a single space.
    :param s: String from which to fix spaces.
    :return: String with spaces fixed.
    """
    while s.find("  ") != -1:
        s = s.replace("  ", " ")
    return s


def remove_special(s, ignore):
    """
    Removes non-alphanumeric characters from a given string, but does not remove characters in ignore, spaces, and
    apostrophes/single quotes [']. Also cleans up bad formatting like trailing whitespaces, double spaces, etc.
    :param s: String from which to remove special characters.
    :param ignore: List of characters to not remove. By default, this function will not remove single spaces.
    :return: String with special characters removed.
    """
    pattern = "[^a-zA-Z0-9' "
    for mark in ignore:
        if mark == '.':
            pattern += '\.'
        elif mark == '?':
            pattern += '\?'
        else:
            pattern += mark
    pattern += ']'
    target = list(re.finditer(pattern, s))
    target = [item.group() for item in target]
    target = set(target)
    edited = s
    for t in target:
        edited = edited.replace(t, ' ')

    edited = remove_double_spaces(edited)
    edited = edited.strip()

    for mark in ignore:
        while edited.find(' ' + mark) != -1:
            edited = edited.replace(' ' + mark, mark)

    pattern = '[0-9],[0-9]'
    target = list(re.finditer(pattern, edited))
    target = [item.group() for item in target]
    target = set(target)
    for t in target:
        replacement = t.replace(',', '')
        edited = edited.replace(t, replacement)

    return edited
