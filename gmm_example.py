import gmm
import time
import numpy as np
import utils as ut


def example1():
    """
    Full covariance matrices, no observation matrix.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10

    h_train = (rng.standard_normal((n_train, n_dim)) + 1j * rng.standard_normal((n_train, n_dim))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)
    noise_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)
    # the SNR is 0 dB
    y_val = h_val + noise_val

    #
    # GMM training
    #
    tic = time.time()
    gm_full = gmm.Gmm(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='full',
    )
    gm_full.fit(h_train)
    toc = time.time()
    print(f'training done. ({ut.sec2hours(toc-tic)})')

    #
    # GMM evaluation
    #
    tic = time.time()
    h_est = gm_full.estimate_from_y(y_val, 0, n_dim, n_summands_or_proba='all')
    print('NMSE of n_summands_or_proba="all":', ut.nmse(h_est, h_val))
    del h_est
    h_est = gm_full.estimate_from_y(y_val, 0, n_dim, n_summands_or_proba=5)
    print('NMSE of n_summands_or_proba=5:', ut.nmse(h_est, h_val))
    del h_est
    toc = time.time()
    print(f'example 1 estimation done. ({ut.sec2hours(toc-tic)})')


def example2():
    import random
    """
    Full covariance matrices with selection matrix A.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10
    n_dim_obs = 5

    # create random selection matrix
    A = np.zeros([n_dim_obs, n_dim])
    pattern_vec = random.sample(range(n_dim), n_dim_obs)
    pattern_vec.sort()
    for i, val in enumerate(pattern_vec):
        A[i, val] = 1

    h_train = (rng.standard_normal((n_train, n_dim)) + 1j * rng.standard_normal((n_train, n_dim))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)
    noise_val = (rng.standard_normal((n_val, n_dim_obs)) + 1j * rng.standard_normal((n_val, n_dim_obs))) / np.sqrt(2)
    # the SNR is 0 dB
    y_val = np.squeeze(np.matmul(A, np.expand_dims(h_val, 2))) + noise_val

    #
    # GMM training
    #
    tic = time.time()
    gm_full = gmm.Gmm(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='full',
    )
    gm_full.fit(h_train)
    toc = time.time()
    print(f'training done. ({ut.sec2hours(toc - tic)})')

    #
    # GMM evaluation
    #
    tic = time.time()
    h_est = gm_full.estimate_from_y(y_val, 0, n_dim, A=A, n_summands_or_proba='all')
    print('NMSE of n_summands_or_proba="all":', ut.nmse(h_est, h_val))
    del h_est
    h_est = gm_full.estimate_from_y(y_val, 0, n_dim, A=A, n_summands_or_proba=5)
    print('NMSE of n_summands_or_proba=5:', ut.nmse(h_est, h_val))
    del h_est
    toc = time.time()
    print(f'example 2 estimation done. ({ut.sec2hours(toc - tic)})')


def example3():
    """
    Diagonal covariance matrices, no observation matrix.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10

    h_train = (rng.standard_normal((n_train, n_dim)) + 1j * rng.standard_normal((n_train, n_dim))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)
    noise_val = (rng.standard_normal((n_val, n_dim)) + 1j * rng.standard_normal((n_val, n_dim))) / np.sqrt(2)
    # the SNR is 0 dB
    y_val = h_val + noise_val

    #
    # GMM training
    #
    tic = time.time()
    gm_diag = gmm.Gmm(
        n_components=16,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='diag',
    )
    gm_diag.fit(h_train)
    toc = time.time()
    print(f'training done. ({ut.sec2hours(toc-tic)})')

    #
    # GMM evaluation
    #
    tic = time.time()
    h_est = gm_diag.estimate_from_y(y_val, 0, n_dim, n_summands_or_proba='all')
    print('NMSE of n_summands_or_proba="all":', ut.nmse(h_est, h_val))
    del h_est
    h_est = gm_diag.estimate_from_y(y_val, 0, n_dim, n_summands_or_proba=5)
    print('NMSE of n_summands_or_proba=5:', ut.nmse(h_est, h_val))
    del h_est
    toc = time.time()
    print(f'example 3 estimation done. ({ut.sec2hours(toc-tic)})')


def example4():
    """
    Full covariance matrices with vectorized MIMO channels and pilot matrix A.
    """
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_rx = 8
    n_tx = 4
    n_components_rx = 8
    n_components_tx = 4

    # mimo channels
    h_train = (rng.standard_normal((n_train, n_rx, n_tx)) + 1j * rng.standard_normal((n_train, n_rx, n_tx))) / np.sqrt(2)
    h_val = (rng.standard_normal((n_val, n_rx, n_tx)) + 1j * rng.standard_normal((n_val, n_rx, n_tx))) / np.sqrt(2)
    # vectorized mimo channels
    channels_train_vec = np.reshape(h_train, (-1, n_rx * n_tx), 'F')
    channels_val_vec = np.reshape(h_val, (-1, n_rx * n_tx), 'F')
    # split mimo channels into rx and tx part
    channels_rx = np.reshape(np.transpose(h_train,[0,2,1]), (n_tx * n_train, -1), 'F')
    channels_tx = np.reshape(h_train, (n_rx * n_train, -1), 'F')

    # DFT pilot matrix A
    A = np.fft.fft(np.eye(n_tx))
    A = np.sqrt(1 / (n_tx)) * A[:n_tx, :]
    A = np.kron(A.T, np.eye(n_rx))

    # observations y for snr=0dB
    snr = 0.0
    y_val = np.squeeze(np.matmul(A, np.expand_dims(channels_val_vec, 2)))
    y_val = y_val + 10 ** (-snr / 20) * ut.crandn(*y_val.shape)

    #
    # GMM training
    #
    tic = time.time()
    # fit two gmm for rx and tx
    gmm_rx = gmm.Gmm(
        n_components=n_components_rx,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='full',
    )
    gmm_rx.fit(channels_rx)

    gmm_tx = gmm.Gmm(
        n_components=n_components_tx,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='full',
    )
    gmm_tx.fit(channels_tx)

    # create kronecker gmm for estimating vectorized mimo channels
    gmm_kron = gmm.create_mimo_gmm(channels_train_vec, gmm_rx, gmm_tx, n_rx, n_tx, n_components_rx, n_components_tx)

    toc = time.time()
    print(f'training done. ({ut.sec2hours(toc - tic)})')

    #
    # GMM evaluation
    #
    tic = time.time()

    h_est = gmm_kron.estimate_from_y(y_val, snr, n_rx*n_tx, A=A, n_summands_or_proba='all')
    print('NMSE of n_summands_or_proba="all":', ut.nmse(h_est, channels_val_vec))
    del h_est
    h_est = gmm_kron.estimate_from_y(y_val, snr, n_rx*n_tx, A=A, n_summands_or_proba=5)
    print('NMSE of n_summands_or_proba=5:', ut.nmse(h_est, channels_val_vec))
    del h_est
    toc = time.time()
    print(f'example 4 estimation done. ({ut.sec2hours(toc - tic)})')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nr', type=int, default=1)
    parargs = parser.parse_args()

    if parargs.nr == 1:
        example1()
    if parargs.nr == 2:
        example2()
    if parargs.nr == 3:
        example3()
    if parargs.nr == 4:
        example4()
    print("hello")