import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import simba.Framework as fw
import numpy as np
import argparse


def rotation_matrix(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-1 * np.sin(theta), 0, np.cos(theta)],
        ]
    )


def parse_arguments(argv=None):
    """Read the command line.

    :param argv: Arguments to parse, defaulting to ``sys.argv``
    :returns: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Check YAML lattice files for errors.")
    parser.add_argument("filename", help="Lattice definition file")
    parser.add_argument(
        "-d", "--decimals", help="Number of decimals to round", default=4, type=int
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Load a settings file and check the lattice it describes.

    :param argv: Arguments to parse, defaulting to ``sys.argv``
    """
    args = parse_arguments(argv)
    lattice = fw.Framework(None)
    lattice.loadSettings(args.filename)
    lattice.check_lattice(decimals=args.decimals)


if __name__ == "__main__":
    main()
