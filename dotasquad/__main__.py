#!/usr/bin/env python3

import argparse

from . import train

def main():
    """
    Define command line arguments. 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--radiant",
            nargs='*',
            action="store",
            default=[],
            type=str,
            help="radiant picks"
            )
    parser.add_argument(
            "--dire",
            nargs='*',
            action="store",
            default=[],
            type=str,
            help="dire picks"
            )

    args = parser.parse_args()
    args_dict = vars(args)

    radiant = args_dict['radiant']
    dire = args_dict['dire']

    print(radiant)
    print(dire)

if __name__ == "__main__":
    main()
