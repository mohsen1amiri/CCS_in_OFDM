# coding: utf-8

import subprocess as sb
import numpy as np
import pandas as pd
import os

import problems
import utils

####################################################
## DATA GENERATION/LOADING                        ##
####################################################

data_list = ["linear", "logistic", "classic", "ofdm"]
simulated_list = {
    "linear": (lambda x, theta : np.dot(np.transpose(x), theta)), 
    "logistic": (lambda x, theta : 1/(1+np.exp(-np.dot(x.T, theta)))),
    }

## build ith canonical vector from R^b
def e(i, b):
    assert i <= b
    e_i = [0]*b
    e_i[i-1] = 1
    return np.matrix([e_i]).T

def import_df(fname):
    df = pd.read_csv(fname)
    df.index = list(map(str, df[df.columns[0]]))
    df = df.loc[~df.index.duplicated()]
    df = df.drop(columns=[df.columns[0]])
    return df

def same_order_df(ids, df_list, axis_list):
    assert len(df_list) == len(axis_list)
    ids_ = list([idx for idx in ids if all([idx in df.columns for df in df_list])])
    assert len(ids_) > 0
    df_res_list = [None]*len(df_list)
    for sdf, df in enumerate(df_list):
        assert axis_list[sdf] in range(2)
        if (axis_list[sdf] == 0):
            df_res_list[sdf] = df.loc[ids_]
        else:
            df_res_list[sdf] = df[ids_]
    return df_res_list


#################################
## NEW - OFDM feature construction   ##
#################################
def build_ofdm_features(args, N, K, seed=0,
                        snr_lin_mu=0.0):
    """
    Returns F in R^{d x N} where each column f_i is a feature vector for subcarrier i.
    dim suggestions:
      - f1: E[|h_i|] or long-term magnitude proxy
      - f2: Var[|h_i|] proxy (or a second slow stat)
      - f3: normalized frequency index i/N
      - f4: recent mean SNR (dB) proxy
      - f5..: extra engineered dims or random small noise features
    """
    # rng = np.random.default_rng(seed)

    # # frequency index feature (shape (K,))
    # idx_feat = np.linspace(0, 1, K, endpoint=False)

    # # slow SNR statistics per-subcarrier in dB (proxy for large-scale)
    # snr_db = rng.normal(loc=snr_mu_db, scale=snr_std_db, size=K)

    # # convert to linear scale if you ever need it elsewhere:
    # # snr_lin = 10**(snr_db/10.0)

    # # base two stats features
    # f1 = np.abs(h)      # ~ E[|h_i|] proxy
    # f2 = np.abs(rng.normal(loc=0.3, scale=0.1, size=K))      # ~ Var[|h_i|] proxy
    # f3 = idx_feat                                           # normalized frequency
    # f4 = snr_db / 20.0                                      # scaled SNR(dB) proxy

    # # stack and then pad to desired d with small random dims (if needed)
    # F0 = np.vstack([f1, f2, f3, f4])                        # shape (4, N)
    # if N > F0.shape[0]:
    #     Fpad = rng.normal(loc=0.0, scale=0.1, size=(N - F0.shape[0], K))
    #     F = np.vstack([F0, Fpad])
    # else:
    #     F = F0[:N, :]

    # # column-normalize gently so no single dim dominates (optional)
    # col_norms = np.linalg.norm(F, axis=0) + 1e-12
    # F = F / col_norms

 

    rng = np.random.default_rng(seed)

    # frequency index feature (shape (K,))

    # slow SNR statistics per-subcarrier in dB (proxy for large-scale)
    snr_lin = np.array(snr_lin_mu)
    # print("snr_lin", snr_lin**0)

    # convert to linear scale if you ever need it elsewhere:
    # snr_lin = 10**(snr_db/10.0)

    F = np.vstack([snr_lin**0])   
    for i in range(N-1):
        F = np.vstack([F, snr_lin**(i+1)])

    # column-normalize gently so no single dim dominates (optional)
    col_norms = np.linalg.norm(F, axis=0) + 1e-12
    F = F / col_norms

    return np.matrix(F)  # shape (N, K)


# def _sumrate_for_subset(snr_lin, subset_idx):
#     """Shannon sum-rate (bits/s/Hz) over a subset of subcarriers."""
#     snr_sel = snr_lin[subset_idx]
#     return float(np.sum(np.log2(1.0 + snr_sel)))

def _sample_channel_gains(K, channel="rayleigh", seed=0, K_factor_dB=6.0):
    """
    Sample complex channel gains h_i (size K).
    - rayleigh: CN(0,1)  (E[|h|^2]=1)
    - rician:  K-factor in dB; h = LOS + NLOS, normalized so E[|h|^2]=1
    """

    rng = np.random.default_rng(seed)
    if channel.lower() == "rayleigh":
        # CN(0,1): real/imag ~ N(0, 0.5)
        re = rng.normal(0.0, 1.0/np.sqrt(2.0), size=K)
        im = rng.normal(0.0, 1.0/np.sqrt(2.0), size=K)
        h = re + 1j*im
        return h
    elif channel.lower() == "rician":
        K_lin = 10**(K_factor_dB/10.0)
        s = np.sqrt(K_lin/(K_lin+1.0))           # LOS power share
        sigma = 1.0/np.sqrt(2.0*(K_lin+1.0))     # NLOS std so E[|h|^2]=1
        re = rng.normal(0.0, sigma, size=K)
        im = rng.normal(0.0, sigma, size=K)
        # LOS on real axis (phase 0) — adjust if you want random LOS phase
        h = s + (re + 1j*im)
        return h
    else:
        raise ValueError(f"Unknown channel model: {channel}")



def apply_snr_noise_db_gaussian(snr_lin, sigma_dB, seed=0):
    """
    Multiplicative lognormal noise on linear SNR from Gaussian dB error.
    snr_lin : np.ndarray (true linear SNRs)
    sigma_dB: float, std-dev of SNR estimation error in dB (e.g., 0.5–2.0)
    rng     : np.random.Generator for reproducibility (optional)
    """
    rng = np.random.default_rng(seed)
    eps_dB = rng.normal(loc=0.0, scale=float(sigma_dB), size=snr_lin.shape)
    scale = 10.0 ** (eps_dB / 10.0)          # lognormal multiplicative factor
    snr_hat = snr_lin * scale
    return snr_hat


def build_ofdm_reward_generator(args, h, seed=0):

    # Channel gains h_i
    

    # Power and noise:
    # - If args.P is given: that's per-subcarrier power P
    # - Else if args.Ptot is given: P = Ptot / m (equal split across active subcarriers)
    # - Else default P=1.0
    if hasattr(args, "P") and args.P is not None:
        P = float(args.P)
    elif hasattr(args, "Ptot") and args.Ptot is not None:
        P = float(args.Ptot) / max(1, args.m)
    else:
        P = 1.67e-3  # 0 dBm = 1 mW over 600 subcarriers = 1.67e-3 mW per subcarrier
                # --- Channel/SNR per the paper ---
                

    sigma2 = float(getattr(args, "sigma2", 1.89e-16))  # -127 dBm/Hz * 15 kHz = -109.2 dBm = 1.89e-16 mW

    # SNR per subcarrier: gamma_i = |h_i|^2 * P / sigma^2
    PL = -120 # path loss in dB
    PL_lin = 10**(PL/10.0)
    P = P * PL_lin
    abs_h_sq = np.abs(h)**2
    snr_lin = (abs_h_sq * P) / (sigma2 + 0.0)
    # --- Option B: Gaussian error in dB ---
    sigma_dB = float(getattr(args, "sigma", 0.2))      # tune this
    snr_lin_hat = apply_snr_noise_db_gaussian(snr_lin, sigma_dB, seed=seed)



    # snr_vec: (N,), S: iterable indices
    # returns sum log2(1 + SNR_i)
    # scores = snr_lin
    scores = np.log2(1.0 + snr_lin)
    return scores.flatten().tolist(), snr_lin_hat.flatten().tolist()

#' @param args Python dictionary of strings
#' @param folder_path Python character string
#' @param normalized Python bool
def create_scores_features(args, folder_path, normalized=False):
    '''Compute/retrieve feature matrix + normalized, if @normalized set to True, "oracle" scores from DR data @data'''
    data = args.data
    #assert utils.is_of_type(data, "str")
    #assert utils.is_of_type(normalized, "bool")
    #################################
    ## Using drug repurposing data ##
    #################################
    if (not (data in data_list)):
        assert not (utils.is_of_type(data, "NoneType"))
        assert data == "epilepsy"
        from constants import dr_folder
        ## Not considering the toy DR problem with 10 arms in the paper
        if (args.small_K != 10):
            ## Arm features
            X = import_df(dr_folder+data+"_signatures_nonbinarized.csv")
            ## Signatures that will be used to compute phenotypes through GRN
            ## S := binarize(X)
            S = import_df(dr_folder+data+"_signatures_binarized.csv")
            ## "True" drug scores
            A = import_df(dr_folder+data+"_scores.csv")
            ## Ordered by drug signature ids
            A, X, S = same_order_df(list(X.columns), [A, X, S], [0, 1, 1])
            names = list(A["drug_name"])
            scores = list(map(float, A["score"]))
            df_di = {"S": S, "X": X, "names": names}
            X = np.matrix(X.values)
        ## Subset of drugs where rewards were pre-recorded
        else:
            file_="rewards_cosine_10drugs_18samples"
            file_features="epilepsy_signatures.csv"
            ## Known anti-epileptics
            names = ["Hydroxyzine", "Acetazolamide", "Pentobarbital", "Topiramate", "Diazepam"]
            ## Known pro-convulsants
            names += ["Dmcm", "Brucine", "Fipronil", "Flumazenil", "Fg-7142"]
            assert len(names) == 10
            drug_ids, drug_positions = utils.get_drug_id(names, dr_folder+file_+".txt")
            assert not any([str(s) == "None" for s in drug_ids])
            A = import_df(dr_folder+data+"_scores.csv")
            drug_cids = A.index
            A.index = A["drug_name"]
            A["drug_cid"] = drug_cids
            drug_cids = list(map(str, A.loc[names]["drug_cid"]))
            assert len(drug_cids) == len(names)
            X = import_df(dr_folder+data+"_signatures.csv")
            S = import_df(dr_folder+data+"_signatures_binarized.csv")
            ## Ordered by drug signature ids
            X, S = same_order_df(drug_cids, [X, S], [1]*2)
            rewards = pd.read_csv(dr_folder+file_+".csv", sep=" ", header=None)
            means = rewards.mean(axis=0).values
            scores = [float(means[i]) for i in drug_positions]
            df_di = {"S": S, "X": X, "names": names}
            X = np.matrix(X.values)
    #################################
    ## "Classic" linear bandit     ##
    #################################
    elif (data == "classic"):
        #assert utils.is_of_type(args.omega, "float")
        print(("Omega = " + str(round(args.omega, 3))))
        assert args.small_K and args.m and args.omega
        if (args.problem == "bernouilli"):
            assert np.cos(args.omega) >= 0
        ## canonical base in R^(K-1), modification from case m=1
        ## arms 1, ..., m have rewards == 1
        ## arm m+1 has reward cos(omega)
        ## arm m+2, ..., K have rewards == 0
        m, K, N, omega = args.m, args.small_K, args.small_K-1, args.omega
        assert m < N
        X = np.matrix(np.eye(N, K))
        X[0,:m] = 1
        X[:,(m+1):] = X[:,m:(K-1)]
        X[:,m] = np.cos(omega)*e(1, N)+np.sin(omega)*e(m+1, N)
        theta = e(1, N)
        scores = simulated_list["linear"](X, theta).flatten().tolist()[0]
    #################################
    ## Using simulated data        ##
    #################################
    ## same setting than the one where complexity constants are compared
    elif (data in list(simulated_list.keys())):
        max_it_gen = 500
        assert args.small_K
        assert args.small_N
        N, K = args.small_N, args.small_K
        matrix_file = folder_path+"generated_matrix_N="+str(N)+"_K="+str(K)+".csv"
        if (not os.path.exists(matrix_file)):
            done = False
            it = 0
            while (not done and it < max_it_gen):
                ## Normalizing the feature matrix
                X = np.matrix(np.random.normal(0, args.vr, (N, K)))
                X /= np.linalg.norm(X, 2)
                done = (np.linalg.matrix_rank(X) >= K)
                it += 1
            if (it == max_it_gen):
                print(("Det value: "+str(np.linalg.det(np.dot(X.T, X)))))
                print("Got unlucky...")
            np.savetxt(matrix_file, X)

        else:
            X = np.matrix(np.loadtxt(matrix_file), dtype=float)
        theta = e(1, N)
        scores = simulated_list[data](X, theta).flatten().tolist()[0]

    elif data == "ofdm":


        assert args.small_K
        assert args.small_N
        N, K = args.small_N, args.small_K
        matrix_file = folder_path+"generated_matrix_N="+str(N)+"_K="+str(K)+".csv"

        h = _sample_channel_gains(K, channel=getattr(args, "channel", "rayleigh"), seed=getattr(args, "seed", 0), K_factor_dB=getattr(args, "K_factor_dB", 6.0))
        scores, snr_lin_hat = build_ofdm_reward_generator(args, h, seed=getattr(args, "seed", 0))

        F = build_ofdm_features(
                        args, N=N, K=K,
                        seed=getattr(args, "seed", 0),
                        snr_lin_mu=snr_lin_hat
                    )
        
        # the normalized feature matrix
        X = F  # (N x K)

        # print("size(X)", X.shape)

        if (not os.path.exists(matrix_file)):
            np.savetxt(matrix_file, X)
        else:
            X = np.matrix(np.loadtxt(matrix_file), dtype=float)
                # ---------- NEW: OFDM branch ----------

        

        rng = np.random.default_rng(getattr(args, "seed", 0))
        theta = rng.normal(size=(N, 1))
        theta = theta / (np.linalg.norm(theta) + 1e-12)
        # ---------- OFDM branch (paper-accurate SNR) ----------
        
        # ---------- END OFDM branch ----------

        

        names = None
                # df_di = {
                #         "X": pd.DataFrame(np.asarray(F),
                #                         index=[f"f{j+1}" for j in range(N)],
                #                         columns=[str(i) for i in range(K)]),
                #         "N_subc": K,
                #         "N_features": N,
                #     }
        df_di = {}
                # ---------- END NEW OFDM branch ----------
    else:
        print("Data type not found!")
        raise ValueError
    if (not data in list(simulated_list.keys())):
        ## Linear regression to find the "true" theta
        theta = np.linalg.inv(X.dot(X.T)).dot(X.dot(np.array(scores).T).T)
        ## residual np.linalg.norm(X.T.dot(theta)-scores, 2)
        theta_file = folder_path+data+"_theta_K="+str(args.small_K)+".csv"
        np.savetxt(theta_file, theta)
    if (data in data_list):
        names = None
        df_di = {}
    assert theta.size == np.shape(X)[0]
    if (data in list(simulated_list.keys()) or data in ["classic"]):
        assert len(scores) == args.small_K 
    ## If Bernouilli arms: means must belong to [0,1]
    if (args.problem == "bernouilli" and data in list(simulated_list.keys()) and data != "classic"):
        X = np.matrix(np.random.normal(0.5, 0.5, (args.small_N, args.small_K)))
        X /= np.linalg.norm(X, 2)
        theta = np.matrix(np.random.normal(0.5, 0.5, (args.small_N, 1)))
        theta /= np.linalg.norm(theta, 2)
        scores = list(map(float, theta.T.dot(X).tolist()[0]))
        assert np.all(np.array(scores) >= 0) and np.all(np.array(scores) <= 1)
    if (data == "linear"):
        ## Print Boolean test on complexity constants
        from compare_complexity_constants import compute_H_UGapE, compute_H_optimized_LinGapE
        H_LinGapE = compute_H_optimized_LinGapE(theta, X, args.epsilon, args.m)
        H_UGapE = compute_H_UGapE(theta, X, args.epsilon, args.m)
        print(("Is H_LinGapE < 2*H_UGapE? : "+str(H_LinGapE < 2*H_UGapE)))
        with open(folder_path+data+"_boolean_test_UGapE_LinGapE_"+data+"N="+str(N)+"_K="+str(K)+"_m="+str(args.m)+".txt", "w+") as f:
            s_ = ["H_LinGapE = "+str(H_LinGapE)]
            s_.append("H_UGapE = "+str(H_UGapE))
            s_.append("Is H_LinGapE < 2*H_UGapE? : "+str(H_LinGapE < 2*H_UGapE))
            f.write("\n".join(s_))
    return X, scores, theta, names, df_di
