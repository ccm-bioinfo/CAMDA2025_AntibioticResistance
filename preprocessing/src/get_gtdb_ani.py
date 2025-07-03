#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read Sourmash signature files from stdin, get the closest reference genome
for each input from GTDB using ANI, and print a CSV file of four columns:
'input', 'reference_genome', 'reference_taxonomy', and 'ani'.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from datetime import datetime as dt
from urllib.request import urlretrieve

import sourmash     # 4.9.3


sbt: sourmash.sbt.SBT


def log(message: str):
    """Logs message to stderr with current datetime."""

    print(f"[{dt.now()}] {message}", file=sys.stderr)


def parse_args() -> tuple[list[str], str]:
    """Parse CLI arguments.
    
    Returns
    -------
    files : list of str
        List of Sourmash signature filenames.
    database : str
        Path to GTDB index file.
    """

    try:
        width = min(80, os.get_terminal_size().columns)
    except OSError:
        width = None
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, width=width, max_help_position=40
        )
    )
    parser.epilog = (
        f"example: {parser.prog} -d gtdb.sbt.zip <<< "
        "genomes/ENA_SAMEA800315.fa.gz"
    )
    parser.add_argument(
        "-d", "--database", default="gtdb.sbt.zip", metavar="path",
        help=(
            "path to GTDB index file; if one is not found in the specified "
            "location, it will be automatically downloaded for you (default: "
            "'gtdb.sbt.zip')"
        )
    )

    args = parser.parse_args()
    if sys.stdin.isatty():
        parser.print_help()
        sys.exit(1)
    files: list[str] = [sig.strip() for sig in sys.stdin]
    database: str = args.database

    return files, database


def get_best_ani(
    sig: sourmash.signature.SourmashSignature
) -> tuple[
    sourmash.signature.SourmashSignature,
    sourmash.signature.SourmashSignature | None,
    float
]:
    """Given a signature file, return the closest signature from the global SBT
    object with their average nucleotide identity.
    
    Parameters
    ----------
    sig : sourmash.signature.SourmashSignature
        Path to a Sourmash signature file.
    
    Returns
    -------
    input_sig : sourmash.signature.SourmashSignature
        Same as input.
    closest_sig : sourmash.signature.SourmashSignature or None
        Closest signature to the input signature.
    ani : float
        Average nucleotide identity between the two signatures.
    """

    result = sbt.best_containment(sig)
    if result is None:
        return sig, None, float("nan")
    else:
        return sig, result.signature, sig.avg_containment_ani(result.signature)


def main() -> int:
    """Driver code."""

    global sbt
    processes = int(os.environ.get("CPUS", 10))
    files, database = parse_args()
    signatures = map(next, map(sourmash.load_file_as_signatures, files))
    url = "https://farm.cse.ucdavis.edu/~ctbrown/sourmash-db/gtdb-rs214/gtdb-rs214-reps.k31.sbt.zip"

    if not os.path.exists(database):
        log(
            "WARNING: downloading GTDB index file as it was not found in "
            f"'{database}'"
        )
        urlretrieve(url, database)

    log(f"INFO: loading '{database}' to memory")
    sbt = sourmash.load_file_as_index(database)
    log(f"INFO: started ANI calculation using '{database}' as reference")

    writer = csv.DictWriter(sys.stdout, fieldnames=[
        "input", "reference_genome", "reference_taxonomy", "ani"
    ])
    writer.writeheader()

    with mp.Pool(processes) as pool:
        for sig, closest, ani in pool.imap_unordered(get_best_ani, signatures):
            if closest is None:
                genome = None
                taxon = None
            else:
                genome, taxon = closest.name.split(" ", maxsplit=1) # type: ignore

            writer.writerow({
                "input": f"{sig}", "reference_genome": genome,
                "reference_taxonomy": taxon, "ani": ani
            })
            log(f"INFO: finished with {sig}")

    log(f"INFO: finished ANI calculation using '{database}' as reference")

    return 0


if __name__ == "__main__":
    sys.exit(main())
