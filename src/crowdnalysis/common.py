"""Module for shared functionality"""


def vprint(*args, verbose: bool = True):
    """Verbose print. Prints only when `verbose` parameter is True"""
    if verbose:
        print(*args)
    return None
