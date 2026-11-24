"""Convert an MP2 processed cohort into MP1 inputs for Run A and Run B. See mp2_cohort.py."""
import argparse
import json

from mp2_cohort import convert_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-dir', required=True, help='MP2 processed cohort directory, for example .../merged_v001')
    parser.add_argument('--out-dir', required=True, help='output root. Writes inputs/ and run_a_repo/ under it')
    parser.add_argument('--limit', type=int, default=None, help='keep only the first N samples (pilot)')
    args = parser.parse_args()
    manifest = convert_cohort(args.cohort_dir, args.out_dir, limit=args.limit)
    summary = {k: manifest[k] for k in ('n_samples', 'n_full_coverage', 'covariate_missing_counts',
                                        'mut_profiled_fraction', 'cna_profiled_fraction')}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
