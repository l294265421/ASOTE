# -*- coding: utf-8 -*-


import argparse


def my_bool(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')