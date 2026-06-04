import argparse
import os, time, sys, random
from generate_embeddings import *
from transfer_learn import *
from use_transfer_learned_model import *

def predict_using_MutationProjector():
    #############################################
    ## User inputs
    #############################################
    parser = argparse.ArgumentParser(description='Use a pretrained MutationProjector for downstream tasks')
    # arguments for generating embeddings
    parser.add_argument('-downstream_train', help='name of the downstream dataset to additionally train. Ignored if using transfer learned model', type=str, default='na')
    parser.add_argument('-transfer_learned_model', help="use transfer learned model. Options include: 'Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC'.", type=str, default='na')
    parser.add_argument('-downstream_eval', help='name of the downstream dataset to predict', type=str)
    parser.add_argument('-padding_idx', help='List of indices for missing values in covariates', nargs='*', default=[], type=int)
    parser.add_argument('-cuda_device', help='cuda device', type=int, default=0)
    # arguments for downstream task
    parser.add_argument('-max_depth', help='max_depth for random forest', type=int, default=10)
    parser.add_argument('-n_estimators', help='n_estimators for random forest', type=int, default=100)
    parser.add_argument('-random_state', help='random_state for Reproducibility', type=int, default=42)
    # output file
    parser.add_argument('-o', help='name for the output prediction result file', type=str, default=None)
    # optional arguments (path to inputs/outputs)
    parser.add_argument('-path_train', help='[optional] path to the folder containing training data. Will override <downstream_train>.', default=None)
    parser.add_argument('-path_test', help='[optional] path to the folder containing test data. Will override <downtream_eval>.', default=None)

    # args
    args = parser.parse_args()

    print(f'Started Running MutationProjector, {time.ctime()}')
    

    ## inputs
    pretrained_model = "pretrained_model.pth"
    geneset          = "MSKIMPACT468"
    networks         = "GRN;E3;phosphorylation;physical_ppi;genetic_interaction;DDRAM;STRING;PCNET"
    split_train_data = 0
    num_features     = 10
    num_GATblock     = 2 
    dff              = 10
    use_rep          = 1
    use_pooling      = 0
    use_gradclip     = 0
    use_covariates   = 1
    num_bins         = 5
    epoch            = 100
    lr               = 0.001
    dropout_p        = 0
    num_heads        = 1
    mask_percentage  = 0
    batch_size       = 64
    weight_decay     = 0.0001


    #############################################
    ## generate embeddings
    #############################################
    # downstream train
    if args.transfer_learned_model == 'na':
        assert (args.downstream_train != 'na') or (args.path_train != None), "provide 'downstream_train' or 'path_train' info"
        embed_from_pretrained(pretrained_model, args.downstream_train, 'train_dataset', geneset, networks, args.padding_idx, split_train_data, num_features, num_GATblock, dff, use_rep, use_pooling, use_gradclip, use_covariates, num_bins, epoch, args.cuda_device, lr, dropout_p, num_heads, mask_percentage, batch_size, weight_decay, args.path_train)
    # use transfer learned model
    else:
        avail_options = ['Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC']
        assert args.transfer_learned_model in avail_options, "'transfer_learned_model' should be one of the following: 'Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC'"
        
    # downstream eval
    assert (args.downstream_eval != 'na') or (args.path_test != None), "provide 'downstream_eval' or 'path_test' info"
    embed_from_pretrained(pretrained_model, args.downstream_eval, 'eval_dataset', geneset, networks, args.padding_idx, split_train_data, num_features, num_GATblock, dff, use_rep, use_pooling, use_gradclip, use_covariates, num_bins, epoch, args.cuda_device, lr, dropout_p, num_heads, mask_percentage, batch_size, weight_decay, args.path_test)
    
    
    #############################################
    ## make predictions
    #############################################
    if args.transfer_learned_model == 'na':
        transfer_learn(args.downstream_train, args.downstream_eval, out_name=args.o, max_depth=args.max_depth, n_estimators=args.n_estimators, random_state=args.random_state, path_train=args.path_train, path_test=args.path_test)
    else:
        use_transfer_learned(args.downstream_eval, args.transfer_learned_model, out_name=args.o, path_test=args.path_test)


if __name__ == '__main__':
    predict_using_MutationProjector()
