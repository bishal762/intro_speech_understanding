def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.
    '''

    month, day = date

    sorted_keys = sorted(birthdays.keys())

    for d in sorted_keys:
        if d > (month, day):
            return d, birthdays[d]

    # wrap around if no later birthday exists
    return sorted_keys[0], birthdays[sorted_keys[0]]
