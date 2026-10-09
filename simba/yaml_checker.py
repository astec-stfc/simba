import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import simba.Framework as fw
import argparse


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
    lattice = fw.Framework(directory=".")
    lattice.loadSettings(args.filename)
    lattice.check_lattice(decimals=args.decimals)


if __name__ == "__main__":
    main()
