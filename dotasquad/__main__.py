#!/usr/bin/env python3

import argparse

from . import opendota
from . import train


def main():
    """
    Define command line arguments. 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--refresh-cache",
            action="store_true",
            default=False,
            help="download latest dota match data"
            )

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict['refresh_cache']:
        opendota.games(refresh=True)

if __name__ == "__main__":
    main()
