"""
Run A (repo recipe): call embed_from_pretrained unchanged on run_a_repo/ text files.
Run from src/, as the repo scripts expect. Outputs go to <dataset-dir>/prediction_results/.
"""
import argparse
import time

from generate_embeddings import embed_from_pretrained

# Clock_SBS1 and Clock_SBS5 (covariate indices 3, 4) have no MP2 source. Pad them with MP1's padding path.
PADDING_IDX = [3, 4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, help='run_a_repo/ directory written by convert_mp2_cohort.py')
    parser.add_argument('--name', required=True, help='dataset name, for example merged_v001')
    args = parser.parse_args()
    print(f'start {time.ctime()}')
    embed_from_pretrained(pretrained_model='pretrained_model.pth', dataset=args.name, dataset_type='eval_dataset',
                          padding_idx=PADDING_IDX, path_dataset=args.dataset_dir)
    print(f'done {time.ctime()}')


if __name__ == '__main__':
    main()
