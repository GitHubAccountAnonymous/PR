def confirm(prompt):
    """
    Asks for user confirmation by typing `y` or `n`.
    :param prompt: Message for user to see.
    :return: True if confirmed, false otherwise.
    """
    res = None
    if not prompt.endswith(' '):
        prompt += ' '
    prompt += 'y or [n]: '
    while res not in ['y', 'n', '']:
        res = input(prompt).lower().strip()
    if res == 'y':
        return True
    else:
        return False