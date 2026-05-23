'''
This homework defines one method, called "arithmetic".
that method, type `help homework2.arithmetic`.
'''

def arithmetic(x, y):
    """
    Modify this code so that it performs one of four possible functions, 
    as specified in the following table:

                        isinstance(x,str)  isinstance(x,float)
    isinstance(y,str)   return x+y         return str(x)+y
    isinstance(y,float) return x*int(y)    return x*y
    """

    if isinstance(x, str):
        if isinstance(y, str):
            return x + y
        else:
            return x * int(y)

    else:
        if isinstance(y, str):
            return str(x) + y
        else:
            return x * y

