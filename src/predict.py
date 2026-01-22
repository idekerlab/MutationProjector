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
    parser.add_argument('-pretrained_model', help='name of the pretrained model (.pth file)', type=str, default='pretrained_model.pth')
    parser.add_argument('-downstream_train', help='name of the downstream dataset to additionally train. Ignored if using transfer learned model', type=str, default='na')
    parser.add_argument('-transfer_learned_model', help="use transfer learned model. Options include: 'Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC'.", type=str, default='na')
    parser.add_argument('-downstream_eval', help='name of the downstream dataset to predict', type=str)
    parser.add_argument('-geneset', help='name of the cancer gene panel', type=str, default='MSKIMPACT468')
    parser.add_argument('-networks', help='list of networks', type=str, default='GRN;E3;phosphorylation;physical_ppi;genetic_interaction;DDRAM;STRING;PCNET')
    parser.add_argument('-padding_idx', help='List of indices for missing values in covariates', nargs='*', default=[], type=int)
    parser.add_argument('-split_train_data', help='do a train/test split for cross validation', type=int, default=0)
    parser.add_argument('-num_features', help='size of the feature embeddings', type=int, default=10)
    parser.add_argument('-num_GATblock', help='number of GAT encoders', type=int, default=2)
    parser.add_argument('-dff', help='size of the feed-forward layer embeddings', type=int, default=10)
    parser.add_argument('-use_rep', help='use representative embeddings for the gene embeddings after GAT encoders', type=int, default=1)
    parser.add_argument('-use_pooling', help='use pooling for the gene embeddings after GAT encoders', type=int, default=0)
    parser.add_argument('-use_gradclip', help='use gradient clipping', type=int, default=0)
    parser.add_argument('-use_covariates', help='use covariates', type=int, default=1)
    parser.add_argument('-num_bins', help='number of bins for TMB and aneuploidy', type=int, default=5)
    parser.add_argument('-epoch', help='epoch used during pre-training', type=int, default=100)
    parser.add_argument('-cuda_device', help='cuda device', type=int, default=0)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('-dropout_p', help='dropout probability', type=float, default=0.1)
    parser.add_argument('-num_heads', help='number of heads per network', type=int, default=1)
    parser.add_argument('-mask_percentage', help='percentage to mask', type=float, default=0)
    parser.add_argument('-batch_size', help='batch size for generating embeddings using pre-trained model', type=int, default=64)
    parser.add_argument('-weight_decay', help='weight decay', type=float, default=0.0001)
    # arguments for downstream task
    parser.add_argument('-max_depth', help='max_depth for random forest', type=int, default=10)
    parser.add_argument('-n_estimators', help='n_estimators for random forest', type=int, default=100)
    parser.add_argument('-random_state', help='random_state for Reproducibility', type=int, default=42)
    # output file
    parser.add_argument('-o', help='name for the output prediction result file', type=str, default=None)
    # args
    args = parser.parse_args()

    print(f'Started Running MutationProjector, {time.ctime()}')
    
    #############################################
    ## generate embeddings
    #############################################
    # downstream train
    if args.transfer_learned_model == 'na':
        assert args.downstream_train != 'na', "provide 'downstream_train' info"
        embed_from_pretrained(args.pretrained_model, args.downstream_train, 'train_dataset', args.geneset, args.networks, args.padding_idx, args.split_train_data, args.num_features, args.num_GATblock, args.dff, args.use_rep, args.use_pooling, args.use_gradclip, args.use_covariates, args.num_bins, args.epoch, args.cuda_device, args.lr, args.dropout_p, args.num_heads, args.mask_percentage, args.batch_size, args.weight_decay)
    # use transfer learned model
    else:
        avail_options = ['Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC']
        assert args.transfer_learned_model in avail_options, "'transfer_learned_model' should be one of the following: 'Chemotherapy', 'Immunotherapy', 'metastasis_luad', 'tissue_of_origin_BRCA', 'tissue_of_origin_COADREAD', 'tissue_of_origin_LUAD', 'tissue_of_origin_LUSC'"
        
    # downstream eval
    embed_from_pretrained(args.pretrained_model, args.downstream_eval, 'eval_dataset', args.geneset, args.networks, args.padding_idx, args.split_train_data, args.num_features, args.num_GATblock, args.dff, args.use_rep, args.use_pooling, args.use_gradclip, args.use_covariates, args.num_bins, args.epoch, args.cuda_device, args.lr, args.dropout_p, args.num_heads, args.mask_percentage, args.batch_size, args.weight_decay)
    
    
    #############################################
    ## make predictions
    #############################################
    if args.transfer_learned_model == 'na':
        transfer_learn(args.downstream_train, args.downstream_eval, out_name=args.o, max_depth=args.max_depth, n_estimators=args.n_estimators, random_state=args.random_state)
    else:
        use_transfer_learned(args.downstream_eval, args.transfer_learned_model, out_name=args.o)


if __name__ == '__main__':
    predict_using_MutationProjector()
