#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read genome IDs from stdin and save them gzipped into a directory. Valid
identifiers include those that start with 'GCA_', 'ENA_' followed by a BioSample
accession or 'BVBRC_' followed by a BV-BRC genome accession. Identifiers may be
separated by any whitespace character."""

import argparse
import csv
import functools as ft
import gzip
import multiprocessing as mp
import os
import subprocess as sp
import sys
import time
from datetime import datetime as dt
from urllib.request import urlopen, urlretrieve


def log(message: str):
    """Logs message to stderr with current datetime."""

    print(f"[{dt.now()}] {message}", file=sys.stderr)


def parse_args() -> tuple[list[str], str]:
    """Parse CLI arguments.
    
    Returns
    -------
    genomes : list of str
        List of genome identifiers.
    output : str
        Output directory.
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
        f"example: {parser.prog} -o genomes <<< "
        "\"ENA_SAMEA800315 BVBRC_485.3331\""
    )
    parser.add_argument(
        "-o", "--output", help="output directory (default: .)",
        default="."
    )
    
    args = parser.parse_args()
    if sys.stdin.isatty():
        parser.print_help()
        sys.exit(1)
    genomes: list[str] = [genome.strip() for genome in sys.stdin.read().split()]
    output: str = args.output

    return genomes, output


def download_genome(genome: str, output: str = ".") -> str:
    """Download a genome gzipped into an output directory.
    
    Parameters
    ----------
    genome : str
        A genome identifier. Valid identifiers include those that start with
        'GCA_', 'ENA_' followed by a BioSample accession, or 'BVBRC_' followed
        by a BV-BRC genome accession.
    output : str, default='.'
        Output directory.
    
    Returns
    -------
    msg : str
        A message useful for logging purposes.
    """

    file = os.path.join(output, f"{genome}.fa.gz")

    if os.path.exists(file):
        msg = f"WARNING: skipping '{genome}' as '{file}' already exists"
    elif genome.startswith("ENA_"):
        biosample = genome.split("_", maxsplit=1)[-1]
        url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={biosample}&result=analysis&fields=analysis_type,submitted_ftp"

        with urlopen(url) as handle:
            contents = handle.read().decode().split("\n")

        reader = csv.reader(contents, delimiter="\t")
        header = next(reader)
        ftp_col = header.index("submitted_ftp")
        type_col = header.index("analysis_type")
        ftp_url = ""

        for row in reader:
            if row[type_col] == "SEQUENCE_ASSEMBLY":
                ftp_url = f"ftp://{row[ftp_col]}"
                break

        if ftp_url:
            try:
                urlretrieve(ftp_url, file)
                msg = f"INFO: finished downloading '{genome}' into '{file}'"
            except Exception as e:
                msg = (
                    f"WARNING: failed to download '{genome}' because of an "
                    f"exception: {e}"
                )
        else:
            msg = (
                f"WARNING: failed to download '{genome}' because no "
                "assembly was found associated with the given BioSample"
            )
        time.sleep(1)

    elif genome.startswith("BVBRC_"):
        identifier = genome.split("_", maxsplit=1)[-1]
        cmd = "p3-get-genome-contigs --attr accession --attr sequence".split()
        stdin = f"genome_id\n{identifier}\n"
        response = sp.run(
            cmd, check=True, capture_output=True, input=stdin, text=True
        ).stdout

        if response == "genome_id\tcontig.accession\tcontig.sequence\n":
            msg = (
                f"WARNING: failed to download '{genome}' as it was not found "
                "in BV-BRC"
            )

        with gzip.open(file, "wt") as handle:
            for i, row in enumerate(response.split("\n")):
                if i == 0 or row == "": continue
                _, contig, sequence = row.split("\t")
                print(
                    f">{identifier}.{contig}\n{sequence.upper()}",
                    file=handle
                )

        time.sleep(1)
        msg = f"INFO: finished downloading '{genome}' into '{file}'"
    elif genome.startswith("GCA_"):
        url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{genome}?download=true&gzip=true"
        try:
            urlretrieve(url, file)
            msg = f"INFO: finished downloading '{genome}' into '{file}'"
        except Exception as e:
            msg = (
                f"WARNING: failed to download '{genome}' because of an "
                f"exception: {e}"
            )
    else:
        msg = f"WARNING: skipping '{genome}' as it is not a valid identifier"

    return msg


def main() -> int:
    """Driver code."""

    processes = int(os.environ.get("CPUS", 10))
    genomes, output = parse_args()
    func = ft.partial(download_genome, output=output)
    log(f"INFO: genome download process into '{output}' started")

    try:
        p3_status = sp.run(
            ["p3-whoami"], check=True, capture_output=True
        ).stdout.decode()
    except FileNotFoundError:
        log("ERROR: PATRIC CLI not found")
        return 1

    if not os.path.exists(output):
        log(f"ERROR: '{output}' does not exist")
        return 1

    if p3_status == "You are currently logged out of PATRIC.\n":
        log("ERROR: you are logged out of PATRIC")
        return 1

    with mp.Pool(processes) as pool:
        for message in pool.imap_unordered(func, genomes):
            log(message)

    log(f"INFO: genome download process into '{output}' finished successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
