from compute_attention import *
from load_model import *
import argparse



# linear probing
def attention_probe(attention_scores, y):
    probe = LinearRegression().fit(attention_scores, y)
    return probe




def compute_importance_():
    #############################################
    ## User inputs
    #############################################
    parser = argparse.ArgumentParser(description='Compute importance')
    # arguments for generating embeddings
    parser.add_argument('-dataset', help='name of the dataset', type=str, default='na')
    parser.add_argument('-dataset_type', help='dataset type ("train_dataset" or "eval_dataset")', type=str, default='eval_dataset')
    # args
    args = parser.parse_args()

    #############################################    
    # fi_dir
    fi_dir = Path().resolve().parent
    
    # out
    out = defaultdict(list) # columns: feature, rho, pval

    # feature info
    features = pd.read_csv(f'{fi_dir}/data/gene/feature_info.txt', sep='\t')
    features = features['feature'].tolist()



    ### load data
    print(f'loading genomic data and prediction results, {time.ctime()}')
    # genomic data
    gData, sData, pData = {}, {}, {}
    # genomic
    mdf = pd.read_csv(f'{fi_dir}/data/downstream_data/{args.dataset_type}/{args.dataset}/mut.txt', sep='\t')
    cna = pd.read_csv(f'{fi_dir}/data/downstream_data/{args.dataset_type}/{args.dataset}/cna.txt', sep='\t')
    cnd = pd.read_csv(f'{fi_dir}/data/downstream_data/{args.dataset_type}/{args.dataset}/cnd.txt', sep='\t')
    merged = merge_data(mdf, cna, cnd, use_cancer_types=False)
    mut = merged[0] 
    # covariates
    cov = pd.read_csv(f'{fi_dir}/data/downstream_data/{args.dataset_type}/{args.dataset}/covariates.txt', sep='\t')
    # pred_out
    assert 'TransferLearning_predictions.txt' in os.listdir(f'{fi_dir}/prediction_results/{args.dataset}'), "'TransferLearning_predictions.txt' file not found."
    pred_out = pd.read_csv(f'{fi_dir}/prediction_results/{args.dataset}/TransferLearning_predictions.txt', sep='\t')
    # model predictions
    y = pred_out['pred_proba'].tolist()

    ### compute attention
    # pretrained_model
    pretrained_model = load_MutationProjector()
    # compute attention
    print(f'computing attention, {time.ctime()}')
    attn, adf = return_attention(mut, cov, pretrained_model)

    ### feature importance
    print(f'computing feature importance, {time.ctime()}')
    for f_idx in range(len(features)):
        # train the probe
        feature = features[f_idx]

        # test data
        attention_in = np.nansum(adf[:,:,f_idx], axis=1)
        attention_out =  np.nansum(adf[:,f_idx], axis=1)
        attention_score = np.stack((attention_in, attention_out), axis=1)
        # gene alterations
        if f_idx < mut.shape[1]:
            additional_features = mut[:,f_idx]
            N_mut = mut[:,f_idx][:,0].sum().item()
            N_cna = mut[:,f_idx][:,1].sum().item()
            N_cnd = mut[:,f_idx][:,2].sum().item()
            all_alt = mut[:,f_idx].sum(axis=1)
            N_alt = sum([1 if all_alt[idx] >= 1 else 0 for idx in range(len(all_alt))])
        # covariates
        else:
            feature_name = feature.replace('MutSig_','')
            additional_features = []
            for sample in pred_out['sample'].tolist():
                additional_features.append(cov.loc[cov['sample']==sample,:][feature_name].item())
            additional_features = np.array(additional_features).reshape(-1,1)
            N_mut, N_cna, N_cnd, N_alt = 'na', 'na', 'na', 'na'

        # probe
        weighted_attention = []
        for sidx in range(len(attention_score)):
            w_att = np.outer(attention_score[sidx], additional_features[sidx]).flatten()
            weighted_attention.append(w_att)
        weighted_attention = np.array(weighted_attention)

        # train a linear probe
        probe = attention_probe(weighted_attention, y)
        probe_predictions = probe.predict(weighted_attention)
        # correlation (predicted responses from full model vs linear probing predictions)
        if len(set(probe_predictions)) == 1:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = stat.spearmanr(probe_predictions, y)
        for key, value in zip(['N_samples', 'feature', 'rho', 'pval', 'N_alt', 'mut', 'cna', 'cnd'],
                              [pred_out.shape[0], feature, rho, pval, N_alt, N_mut, N_cna, N_cnd]):
            out[key].append(value)
    out = pd.DataFrame(out)
    # save results
    out.to_csv(f'{fi_dir}/prediction_results/{args.dataset}/feature_importances.txt', sep='\t', index=False)
    print(f'done, {time.ctime()}')
    
    
    
    
if __name__== '__main__':
    compute_importance_()