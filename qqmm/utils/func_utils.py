def alphanum_key(s: str):
    """Use to sort string by numbers in it, useful when numbers in string are not matched in digit (1 and 001)."""
    digit_key_list = []
    str_section = ''
    is_digit = False
    for c in s:
        if c.isdigit() == is_digit:
            str_section += c
        else:
            if is_digit:
                digit_key_list.append(int(str_section))
            else:
                digit_key_list.append(str_section)
            str_section, is_digit = c, c.isdigit()
    if is_digit:
        digit_key_list.append(int(str_section))
    else:
        digit_key_list.append(str_section)
    return digit_key_list
