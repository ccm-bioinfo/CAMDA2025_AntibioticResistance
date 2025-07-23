#!/usr/bin/env python3

import pandas as pd
import os
import argparse

def analyze_core_families(train_file: str, output_dir: str, threshold_percentage: float = 0.8):
    df = pd.read_csv(train_file)
    bacteria_name = os.path.basename(train_file).replace("pangenome", "").replace("train", "").replace(".csv.bz2", "").strip("_")
    families_df = df.iloc[:, 11:]
    total_genomes = df.shape[0]
    family_freq = (families_df > 0).sum(axis=0)

    frequency_df = pd.DataFrame({
        "family": families_df.columns,
        "genome_frequency": family_freq.values
    })

    freq_dir = os.path.join(output_dir, "frequencies")
    not_core_dir = os.path.join(output_dir, "not_core")
    os.makedirs(freq_dir, exist_ok=True)
    os.makedirs(not_core_dir, exist_ok=True)

    frequency_df.to_csv(os.path.join(freq_dir, f"family_frequency_{bacteria_name}.csv"), index=False)

    threshold = int(threshold_percentage * total_genomes)
    not_core = frequency_df[(frequency_df["genome_frequency"] < threshold) & (frequency_df["genome_frequency"] != 0)]
    core = frequency_df[frequency_df["genome_frequency"] >= threshold]
    zero_families = (frequency_df["genome_frequency"] == 0).sum()

    not_core[["family"]].to_csv(os.path.join(not_core_dir, f"not_core_families_{bacteria_name}.csv"), index=False)

    print(f"Core families (≥ {int(threshold_percentage * 100)}% of genomes): {len(core)}")
    print(f"Non-core families (< {int(threshold_percentage * 100)}%): {len(not_core)}")
    print(f"Families absent in all genomes: {zero_families}")


def filter_non_core_families(train_file: str, test_file: str, output_dir: str):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    bacteria_name = os.path.basename(train_file).replace("pangenome", "").replace("train", "").replace(".csv.bz2", "").strip("_")
    not_core_file = os.path.join(output_dir, "not_core", f"not_core_families_{bacteria_name}.csv")
    families_df = pd.read_csv(not_core_file)
    family_ids = families_df["family"].tolist()

    base_cols_train = list(df_train.columns[:11])
    base_cols_test = list(df_test.columns[:11])
    family_cols_train = [col for col in df_train.columns[11:] if col in family_ids]
    family_cols_test = [col for col in df_test.columns[11:] if col in family_ids]

    df_filtered_train = df_train[base_cols_train + family_cols_train]
    df_filtered_test = df_test[base_cols_test + family_cols_test]

    filtered_dir = os.path.join(output_dir, "pangenomes_filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    df_filtered_train.to_csv(os.path.join(filtered_dir, f"pangenome_{bacteria_name}_train.csv"), index=False)
    df_filtered_test.to_csv(os.path.join(filtered_dir, f"pangenome_{bacteria_name}_test.csv"), index=False)

    print(f"{bacteria_name}: {len(family_cols_train)} non-core families retained (train).")
    print(f"{bacteria_name}: {len(family_cols_test)} non-core families retained (test).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze core/non-core gene families and filter pangenomes."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subcommand: analyze
    parser_analyze = subparsers.add_parser("analyze", help="Identify core and non-core families")
    parser_analyze.add_argument("train_file", help="Path to training pangenome CSV file")
    parser_analyze.add_argument("output_dir", help="Output directory")
    parser_analyze.add_argument("--threshold", type=float, default=0.8, help="Threshold percentage (default: 0.8)")

    # Subcommand: filter
    parser_filter = subparsers.add_parser("filter", help="Filter pangenomes by non-core families")
    parser_filter.add_argument("train_file", help="Path to training pangenome CSV file")
    parser_filter.add_argument("test_file", help="Path to test pangenome CSV file")
    parser_filter.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_core_families(args.train_file, args.output_dir, args.threshold)
    elif args.command == "filter":
        filter_non_core_families(args.train_file, args.test_file, args.output_dir)
    else:
        parser.print_help()
