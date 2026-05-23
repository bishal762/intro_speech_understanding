def words2characters(words):
    """
    This function converts a list of words into a list of characters.
    """
    
    characters = []

    for word in words:
        for char in str(word):
            characters.append(char)

    return characters


def cancellation(input_list, stop_word):

    output_list = []

    for item in input_list:
        if item == stop_word:
            break
        output_list.append(item)

    return output_list


def copy_all_but_skip_word(input_list, skip_word):

    output_list = []

    for item in input_list:
        if item == skip_word:
            continue
        output_list.append(item)

    return output_list


def my_average(input_list):

    total = 0

    for item in input_list:
        total += item

    return total / len(input_list)
