import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
import sdp
import glasso as gl
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}"
})
from tqdm.notebook import tqdm
from scipy.linalg import sqrtm

# TODO: RUN the following settings:
# TODO: nodes vary from 200 to 6000. Increment by 200. 
# TODO: 1. Low rank coherence bounded by n^{-1/4}. Recover Bstar at recovering threshold. (Gaussian noise)
# TODO: 2. Large low rank coherence, without removal, recover Bstar at recovering threshold. (Gaussian noise)

def spectral_initializer(n: int, mu: float = None, r: int = 3, evs: np.ndarray = None):
    if evs is None:
        evs = 2*log(n)*np.arange(1, r + 1)  
        evs += 3*sqrt(n)
    
    if mu is None:
        mu = n**(1/2) * log(n) ** (-1/2)
    
    Mstar, _ = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    W0 = g.symmetric_gaussian_noise_heter(n)
    M = Mstar + W0
    del W0
    
    return (Mstar + g.symmetric_gaussian_noise_heter(n) - lr.low_rank_entrywise(M, r))
    

def exp_low_rank_Bstar(n: int, m: int, method: str = "sdp",
                       r: int = 3, mu: float = None, 
                       scale: float = None,
                       evs: np.ndarray = None, verbose=False):
    
    assert method in ["sdp", "lasso"], "Method must be either 'sdp' or 'lasso'."
    
    if scale is None:
        scale = 2*n**(-1/4)*log(n)**(1/4)
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n, m, scale)
    Ytil = Bstar + spectral_initializer(n, mu=mu, r=r, evs=evs)
     
    if verbose:
        u.heatmap(Ytil, title="observation")
    
    
    if method == "sdp":
        Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
        if verbose:
            u.heatmap(1-Zhat, title="estimate")
        Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    elif method == "lasso":
        Ihat = gl.glasso(Ytil, m, verbose=verbose)
    
    results = {
        "error":  np.sum(~np.isin(Ihat, Istar)),
    }
    
    return results
    




# TODO: 3. Directly working with Bstar + W. Demonstrate Glasso solution path. (Gaussian noise, single copy)

def exp_lasso_path(n: int, m: int, scale: float = None, T: int = 60):
    Bstar, Istar = g.generate_node_sparse_signal_general(n, m, scale)
    Ytil = Bstar + g.symmetric_gaussian_noise_homo(n)
    paths = np.zeros((n, T))
    lambda_max = np.max(np.linalg.norm(Ytil, axis=1))
    lambda_min = 0.85 * np.min(np.linalg.norm(Ytil, axis=1)) 
    lambdas = np.linspace(lambda_min, lambda_max, T)
    t = 0
    for lamda in tqdm(lambdas):
        _, V = gl.ADMM(Ytil, lamda)
        alphas = np.sqrt(np.sum(V**2, axis=0))
        paths[:,t] = alphas
        t += 1
    
    plotted_labels = set()
    plt.figure(figsize=(8, 5)) 
    plt.style.use('seaborn-white')
    
    for i in range(n):
        if i in Istar:
            label = 'Ground Truth' if 'Ground Truth' not in plotted_labels else None
            plt.plot(lambdas, paths[i, :], label=label, color='tab:orange', linewidth=2)
            plt.scatter(lambdas, paths[i, :], color='tab:orange', s=10)
            plotted_labels.add('Ground Truth')
        else:
            if 'Null' not in plotted_labels:
                plt.plot([], [], label='Null', color='tab:blue')  # Dummy for legend
                plotted_labels.add('Null')
            plt.plot(lambdas, paths[i, :], color='tab:blue', alpha=0.2)
    
    x0, y0 = lambda_max, 0
    slope = -1

    # Choose the x-range over which to plot the line
    x_vals = np.linspace(lambda_min, lambda_max, 100)
    y_vals = slope * (x_vals - x0) + y0

    # Add to your existing plot
    plt.plot(x_vals, y_vals, color='tab:red', linestyle='--', alpha=0.7, label='Reference Line 1')
    
    x1, y1 = lambda_min, np.max(paths)
    slope = -1/2
    y_vals = slope * (x_vals - x1) + y1
    plt.plot(x_vals, y_vals, color='tab:red', linestyle=':', alpha=0.7, label='Reference Line 2')

    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel(r'$\bm{\alpha}$', fontsize=14, rotation=0)    
    plt.title('Group Lasso Solution Path', fontsize=16)
    # Move legend outside plot
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)

    # Ticks font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lasso_path.png", facecolor='white', bbox_inches='tight', dpi=300)
    plt.show()



# TODO: 4. Demonstrate failuare of Glasso while SDP works. (Gaussian noise, single copy)
# TODO: 5. Demonstrate that SDP works/fails under more general settings. (Gaussian noise, single copy)


def exp_lasso_failure_Bstar(n: int, m: int, method: str = "sdp",
                            verbose=False):
    
    assert method in ["sdp", "lasso"], "Method must be either 'sdp' or 'lasso'."
    
    Bstar, Istar = g.generate_Bstar_failure(n,m)
    Ytil = Bstar + g.symmetric_gaussian_noise_homo(n)
     
    if verbose:
        u.heatmap(Ytil, title="Failure Observation")
    
    if method == "sdp":
        Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
        Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    elif method == "lasso":
        Ihat = gl.glasso(Ytil, m, verbose=verbose)
    
    results = {
        "error":  np.sum(~np.isin(Ihat, Istar)),
    }
    
    return results




# TODO: 6 Demonstrate failure under heteroskedastic noise. (Gaussian noise, single copy)
# TODO: 7. Demonstrate success under heteroskedastic noise. (Gaussian noise, multiple copy)


def exp_dual_heter(n: int, m: int, verbose=False):
    
    results = {}
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n,m)
    Y0 = Bstar + g.symmetric_gaussian_noise_row_heter(n)
    Y1 = Bstar + g.symmetric_gaussian_noise_row_heter(n)
    
    Zhat = sdp.solve_sdp_mosek(Y0*Y1, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]

    results["multiple_error"] = np.sum(~np.isin(Ihat, Istar))
    
    Y = (Y0 + Y1) / 2
    Zhat = sdp.solve_sdp_mosek(Y*Y, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]

    results["single_error"] = np.sum(~np.isin(Ihat, Istar))
    
    return results


# TODO: 8. Demonstrate truncated SDP under Heavy-tailed noise.

def exp_trunc(n: int, m: int, df: int = 4, verbose: bool = False):
    
    results = {}

    Bstar, Istar = g.generate_node_sparse_signal_general(n,m)
    Ytil = Bstar + g.symmetric_t_noise_homo(n, df)
     
    if verbose:
        u.heatmap(Ytil, title="Heteroskedastic Noisy Observation")
    
    tau = 1
    
    Zhat = sdp.solve_sdp_mosek(np.minimum(Ytil*Ytil, tau**2), m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    results["trunc_error"] = np.sum(~np.isin(Ihat, Istar))
    
    Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    results["sdp_error"] = np.sum(~np.isin(Ihat, Istar))
    
    return results


# TODO: 9. Show improved recovery of the low rank structure for Ustar under Gaussian noise / Heavy-tailed noise. 
# TODO: 10. Show improved recovery of the low rank structure for Mstar under Gaussian noise / Heavy-tailed noise. 

def improved_eigvec(U, W):
    n, r = U.shape
    Uhat = np.zeros((n,r))
    U = np.real(U)
    W = np.real(W)
    for l in range(r):
        UWl = U[:,l].dot(W[:,l])
        Uhat[:,l] = np.sign(U[:,l]) * np.minimum(
            np.sqrt(
                np.abs(U[:,l]*W[:,l]/UWl)
                ), 
            1)
    
    return Uhat
    
def asymm_arrange(M1, M2):
    M = M2
    i_lower = np.tril_indices_from(M1, k=-1)
    M[i_lower] = M1[i_lower]
    return M

def get_Uhat_Mhat_util(eigvals_r, U, M1, M2):
    evs_rinv = np.diag(1/eigvals_r)
    evs_r = np.diag(eigvals_r)
    G = np.linalg.inv(evs_rinv @ U.T @ M1 @ M2 @ U @ evs_rinv)
    Gsymm = (G + G.T) / 2
    Psihat = sqrtm(Gsymm)
    Uhat = U @ Psihat
    Mhat = Uhat @ evs_r @ Uhat.T
    return Uhat, Mhat

def get_Uhat_Mhat(M11, M12, M21, M22, r):
    M = asymm_arrange(M11, M12)
    _, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    Uhat1, Mhat1 = get_Uhat_Mhat_util(eigvals_r, U, M21, M22)
    M = asymm_arrange(M11, M12)
    _, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    Uhat2, Mhat2 = get_Uhat_Mhat_util(eigvals_r, U, M21, M22)
    
    Uhat = (Uhat1 + Uhat2) / 2
    Mhat = (Mhat1 + Mhat2) / 2
    
    return Uhat, Mhat

def get_Uhat_Mhat_base(M11, M12, M21, M22, r):
    M1 = asymm_arrange(M11, M12)
    M2 = asymm_arrange(M21, M22)
    M = (M1 + M2) / 2
    del M1, M2
    
    W, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    evs_r = np.diag(eigvals_r)
    Uhat = improved_eigvec(U, W)
    Mhat = Uhat @ evs_r @ Uhat.T
    return Uhat, Mhat

def exp_low_rank(
        n:int, mu:float, r:int = 2, evs: np.ndarray = None
    ):
    
    results = {}
    
    if evs is None:
        evs = 4*log(n)*np.arange(r)  
        evs += 2.5*sqrt(n)
        
    Mstar, Ustar = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    M11 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M12 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M21 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M22 = Mstar + g.symmetric_gaussian_noise_homo(n)
    
    Uhat, Mhat = get_Uhat_Mhat(M11, M12, M21, M22, r)
    
    # Gamma, _, Rt = np.linalg.svd(U.T @ Ustar, full_matrices=False)
    # O = Gamma @ Rt
    
    results["Uhat_error"] = u.tti_approx(Uhat, Ustar)
    results["Mhat_error"] = u.max_abs(Mhat - Mstar)
    
    Uhat, Mhat = get_Uhat_Mhat_base(M11, M12, M21, M22, r)
    results["Ubase_error"] = u.tti_approx(Uhat, Ustar)
    results["Mbase_error"] = u.max_abs(Mhat - Mstar)

    Mhat, Uhat, _ = lr.top_r_low_rank_symmetric((M11+M12+M21+M22)/4, r)
    results["Uspec_error"] = u.tti_approx(Uhat, Ustar)
    results["Mspec_error"] = u.max_abs(Mhat - Mstar)
    
    return results, Ustar
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    