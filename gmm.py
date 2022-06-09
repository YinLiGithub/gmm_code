import numpy as np
import utils as ut
from scipy import linalg as scilinalg
from sklearn.mixture import GaussianMixture


def compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scilinalg.cholesky(covariance, lower=True)
            except scilinalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = scilinalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = scilinalg.cholesky(covariances, lower=True)
        except scilinalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = scilinalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


def create_mimo_gmm(channels_train_vec, gmm_rx, gmm_tx, n_rx, n_tx, n_components_rx, n_components_tx):
    num_covs = n_components_rx * n_components_tx

    covs_kron_real = np.zeros([num_covs, 2 * n_rx * n_tx, 2 * n_rx * n_tx])
    means_kron_real = np.zeros([num_covs, 2 * n_rx * n_tx])
    means_kron_cplx = np.zeros([num_covs, n_rx * n_tx], dtype=complex)
    covs_rx = gmm_rx.covs.copy()
    # covs_rx_cmplx = ut.cov2cov(covs_rx)
    covs_tx = gmm_tx.covs.copy()
    # covs_tx_cmplx = ut.cov2cov(covs_tx)
    means_rx = gmm_rx.means_cplx
    means_tx = gmm_tx.means_cplx
    it = 0
    for n_r in range(n_components_rx):
        for n_t in range(n_components_tx):
            covs_kron_real[it, :, :] = ut.kron_real(covs_tx[n_t, :, :], covs_rx[n_r, :, :])
            means_kron_cplx[it, :] = np.kron(means_tx[n_t, :], means_rx[n_r, :])
            means_kron_real[it, :] = ut.cplx2real(means_kron_cplx[it, :])
            it += 1
    # covs_cmplx = ut.cov2cov(covs_kron_real)
    # a, b = np.linalg.eig(covs_kron_real[0,:,:])
    chols = compute_precision_cholesky(covs_kron_real, 'full')

    gmm_kron = Gmm(
        n_components=n_components_rx * n_components_tx,
        random_state=2,
        max_iter=100,
        n_init=1,
        covariance_type='full',
    )
    # initialize parameters such that weights are not None
    gmm_kron.gm._initialize_parameters(ut.cplx2real(channels_train_vec, axis=1), random_state=None)

    gmm_kron.gm.precisions_cholesky_ = chols.copy()
    gmm_kron.gm.covariances_ = covs_kron_real.copy()
    gmm_kron.gm.n_features_in_ = 2 * n_rx * n_tx
    gmm_kron.gm.means_ = means_kron_real.copy()
    gmm_kron.means = gmm_kron.gm.means_.copy()
    gmm_kron.means_cplx = means_kron_cplx.copy()
    gmm_kron.covs = covs_kron_real.copy()
    gmm_kron.covs_cplx = ut.cov2cov(covs_kron_real)

    # compute weights by performing single e-step
    n_samples = channels_train_vec.shape[0]
    _, log_resp = gmm_kron.gm._e_step(ut.cplx2real(channels_train_vec, axis=1))
    resp = np.exp(log_resp)
    weights = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    weights /= n_samples
    gmm_kron.gm.weights_ = weights

    return gmm_kron


def mse_func(in1, in2):
    return np.sum(np.abs(in1 - in2)**2) / in1.size


class Gmm():
    def __init__(self, *gmm_args, **gmm_kwargs):
        self.gm = GaussianMixture(*gmm_args, **gmm_kwargs)
        self.means = None
        self.means_cplx = None
        self.covs = None
        self.covs_cplx = None

    def fit(self, h):
        """
        Fit an sklearn Gaussian mixture model using complex data h.
        """
        if self.gm.covariance_type == 'diag':
            h_real = ut.cplx2real(np.fft.fft(h, axis=1) / np.sqrt(h.shape[-1]), axis=1)
            dft_matrix = np.fft.fft(np.eye(h.shape[-1], dtype=np.complex)) / np.sqrt(h.shape[-1])
            self.gm.fit(h_real)
            self.means = self.gm.means_.copy()
            self.means_cplx = ut.real2cplx(self.gm.means_, axis=1) @ dft_matrix.conj()
            self.covs = self.gm.covariances_.copy()
            self.covs_cplx = dft_matrix.conj().T @ ut.cov2cov(self.gm.covariances_) @ dft_matrix
        elif self.gm.covariance_type == 'full':
            h_real = ut.cplx2real(h, axis=1)
            self.gm.fit(h_real)
            self.means = self.gm.means_.copy()
            self.means_cplx = ut.real2cplx(self.gm.means_, axis=1)
            self.covs = self.gm.covariances_.copy()
            self.covs_cplx = ut.cov2cov(self.gm.covariances_)
        else:
            raise NotImplementedError(f'Fitting for covariance_type = {self.gm.covariance_type} is not implemented.')

    def estimate_from_y(self, y, snr_dB, n_antennas, A=None, n_summands_or_proba=1):
        """
        Use the noise covariance matrix and the matrix A to update the
        covariance matrices of the Gaussian mixture model. This GMM is then
        used for channel estimation from y.

        Args:
            y: A 2D complex numpy array.
            snr_dB: The SNR in dB.
            n_antennas: The dimension of the channels.
            A: A complex observation matrix.
            n_summands_or_proba:
                If equal to 'all', compute the sum of all LMMSE estimates.
                If equal to an integer, compute the sum of the top (highest
                    component probabilities) n_summands_or_proba LMMSE
                    estimates.
                If equal to a float, compute the sum of as many LMMSE estimates
                    as are necessary to reach at least a cumulative component
                    probability of n_summands_or_proba.
        """
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        y_for_prediction, covs_Cy_inv = self._prepare_for_prediction(y, A, snr_dB)

        h_est = np.zeros([y.shape[0], A.shape[-1]], dtype=complex)
        if isinstance(n_summands_or_proba, int):
            # n_summands_or_proba represents a number of summands

            if n_summands_or_proba == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self.gm.predict(ut.cplx2real(y_for_prediction, axis=1))
                for yi in range(y.shape[0]):
                    mean_h = self.means_cplx[labels[yi], :]
                    h_est[yi, :] = self._lmmse_formula(
                        y[yi, :], mean_h, self.covs_cplx[labels[yi], :, :] @ A.conj().T, covs_Cy_inv[labels[yi], :, :], A @ mean_h)
            else:
                # use predicted probabilites to compute weighted sum of estimators
                proba = self.gm.predict_proba(ut.cplx2real(y_for_prediction, axis=1))
                for yi in range(y.shape[0]):
                    # indices for probabilites in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        mean_h = self.means_cplx[argproba, :]
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                            y[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A.conj().T, covs_Cy_inv[argproba, :, :], A @ mean_h)
                    h_est[yi, :] /= np.sum(proba[yi, idx_sort[:n_summands_or_proba]])
        elif n_summands_or_proba == 'all':
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.gm.predict_proba(ut.cplx2real(y_for_prediction, axis=1))
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        y[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A.conj().T, covs_Cy_inv[argproba, :, :], A @ mean_h)
        else:
            # n_summands_or_proba represents a probability

            # use predicted probabilites to compute weighted sum of estimators
            proba = self.gm.predict_proba(ut.cplx2real(y_for_prediction, axis=1))
            for yi in range(y.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = np.searchsorted(np.cumsum(proba[yi, idx_sort]), n_summands_or_proba) + 1
                for argproba in idx_sort[:nr_proba]:
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        y[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A.conj().T, covs_Cy_inv[argproba, :, :], A @ mean_h)
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])

        return h_est

    def _prepare_for_prediction(self, y, A, snr_dB):
        """
        Replace the GMM's means and covariance matrices by the means and
        covariance matrices of the observation. Further, in case of diagonal
        matrices, FFT-transform the observation.
        """
        sigma2 = 10 ** (-snr_dB / 10)

        if self.gm.covariance_type == 'diag':
            # raise error if A is not identity or quadratic matrix
            try:
                diff = np.sum(np.abs(A - np.eye(A.shape[0])) ** 2)
                if diff > 1e-12:
                    NotImplementedError(f'Estimation for covariance_type = {self.gm.covariance_type} with arbitrary matrix A is not implemented.')
            except:
                raise NotImplementedError(f'Estimation for covariance_type = {self.gm.covariance_type} with arbitrary matrix A is not implemented.')

            # update GMM covs
            covs_gm = self.covs.copy()
            sigma2_diag = 0.5 * sigma2 * np.ones(covs_gm.shape[-1])
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :] = covs_gm[i, :] + sigma2_diag
            self.gm.covariances_ = covs_gm.copy()      # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(covs_gm, covariance_type='diag')

            # FFT of observation
            y_for_prediction = np.fft.fft(y, axis=1) / np.sqrt(y.shape[-1])
        elif self.gm.covariance_type == 'full':
            # real representation of A and A^H
            A_real = ut.mat2bsc(A)
            A_real_h = ut.mat2bsc(A.conj().T)

            # update GMM means
            Am = np.squeeze(np.matmul(A, np.expand_dims(self.means_cplx, axis=2)))
            # handle the case of only one GMM component
            if Am.ndim == 1:
                self.gm.means_ = ut.cplx2real(Am[None, :], axis=1)
            else:
                self.gm.means_ = ut.cplx2real(Am, axis=1)

            # update GMM covs
            covs_gm = self.covs.copy()
            covs_gm = np.matmul(np.matmul(A_real, covs_gm), A_real_h)
            sigma2_diag = 0.5 * sigma2 * np.eye(covs_gm.shape[-1])
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :, :] = covs_gm[i, :, :] + sigma2_diag
            self.gm.covariances_ = covs_gm.copy()      # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(covs_gm, covariance_type='full')

            # update GMM feature number
            self.gm.n_features_in_ = 2 * A.shape[0]

            y_for_prediction = y
        else:
            raise NotImplementedError(f'Estimation for covariance_type = {self.gm.covariance_type} is not implemented.')

        # precompute the inverse matrices
        cov_noise = sigma2 * np.eye(y.shape[-1], dtype=np.complex)
        covs_Cy_inv = np.zeros([self.covs_cplx.shape[0], A.shape[0], A.shape[0]], dtype=complex)
        for i in range(self.covs_cplx.shape[0]):
            covs_Cy_inv[i, :, :] = np.linalg.pinv(A @ self.covs_cplx[i, :, :] @ A.conj().T + cov_noise)

        return y_for_prediction, covs_Cy_inv

    def _lmmse_formula(self, y, mean_h, cov_h, cov_y_inv, mean_y):
        return mean_h + cov_h @ (cov_y_inv @ (y - mean_y))

    def estimate_from_r(self, r, snr_dB, n_antennas, A=None, n_summands_or_proba=1):
        """
            Use the noise covariance matrix and the matrix A to update the
            covariance matrices of the Gaussian mixture model. This GMM is then
            used for channel estimation from one-bit quantized r = Q(y).

            Args:
                y: A 2D complex numpy array.
                snr_dB: The SNR in dB.
                n_antennas: The dimension of the channels.
                A: A complex observation matrix.
                n_summands_or_proba:
                    If equal to 'all', compute the sum of all LMMSE estimates.
                    If equal to an integer, compute the sum of the top (highest
                        component probabilities) n_summands_or_proba LMMSE
                        estimates.
                    If equal to a float, compute the sum of as many LMMSE estimates
                        as are necessary to reach at least a cumulative component
                        probability of n_summands_or_proba.
            """
        if A is None:
            A = np.eye(n_antennas, dtype=r.dtype)
        r_for_prediction, covs_Cr_inv = self._prepare_for_prediction_1bit(r, A, snr_dB)

        h_est = np.zeros([r.shape[0], A.shape[-1]], dtype=complex)
        if isinstance(n_summands_or_proba, int):
            # n_summands_or_proba represents a number of summands

            if n_summands_or_proba == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self.gm.predict(ut.cplx2real(r_for_prediction, axis=1))
                for yi in range(r.shape[0]):
                    mean_h = self.means_cplx[labels[yi], :]
                    A_eff = np.sqrt(2/np.pi) * self.Psi_12[labels[yi], :, :] @ A
                    h_est[yi, :] = self._lmmse_formula(
                        r[yi, :], mean_h, self.covs_cplx[labels[yi], :, :] @ A_eff.conj().T, covs_Cr_inv[labels[yi], :, :],
                                          A_eff @ mean_h)
            else:
                # use predicted probabilites to compute weighted sum of estimators
                proba = self.gm.predict_proba(ut.cplx2real(r_for_prediction, axis=1))
                for yi in range(r.shape[0]):
                    # indices for probabilites in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        mean_h = self.means_cplx[argproba, :]
                        A_eff = np.sqrt(2 / np.pi) * self.Psi_12[argproba, :, :] @ A
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                            r[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A_eff.conj().T, covs_Cr_inv[argproba, :, :],
                                              A_eff @ mean_h)
                    h_est[yi, :] /= np.sum(proba[yi, idx_sort[:n_summands_or_proba]])
        elif n_summands_or_proba == 'all':
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.gm.predict_proba(ut.cplx2real(r_for_prediction, axis=1))
            for yi in range(r.shape[0]):
                for argproba in range(proba.shape[1]):
                    mean_h = self.means_cplx[argproba, :]
                    A_eff = np.sqrt(2 / np.pi) * self.Psi_12[argproba, :, :] @ A
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        r[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A.conj().T, covs_Cr_inv[argproba, :, :],
                                          A_eff @ mean_h)
        else:
            # n_summands_or_proba represents a probability

            # use predicted probabilites to compute weighted sum of estimators
            proba = self.gm.predict_proba(ut.cplx2real(r_for_prediction, axis=1))
            for yi in range(r.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = np.searchsorted(np.cumsum(proba[yi, idx_sort]), n_summands_or_proba) + 1
                for argproba in idx_sort[:nr_proba]:
                    mean_h = self.means_cplx[argproba, :]
                    A_eff = np.sqrt(2 / np.pi) * self.Psi_12[argproba, :, :] @ A
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(
                        r[yi, :], mean_h, self.covs_cplx[argproba, :, :] @ A_eff.conj().T, covs_Cr_inv[argproba, :, :],
                                          A_eff @ mean_h)
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])

        return h_est


    def _prepare_for_prediction_1bit(self, r, A, snr_dB):
        """
        Replace the GMM's means and covariance matrices by the means and
        covariance matrices of the observation. Further, in case of diagonal
        matrices, FFT-transform the observation.
        """
        sigma2 = 10 ** (-snr_dB / 10)
        if self.gm.covariance_type == 'full':
            # real representation of A and A^H
            A_real = ut.mat2bsc(A)
            A_real_h = ut.mat2bsc(A.conj().T)

            # update GMM covs
            covs_gm = self.covs.copy()
            covs_gm = np.matmul(np.matmul(A_real, covs_gm), A_real_h)
            sigma2_diag = 0.5 * sigma2 * np.eye(covs_gm.shape[-1])
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :, :] = covs_gm[i, :, :] + sigma2_diag
            Cy_cplx = ut.cov2cov(covs_gm)
            self.Psi_12 = np.zeros_like(Cy_cplx)
            for it in range(self.Psi_12.shape[0]):
                self.Psi_12[it,:,:] = np.real(np.diag(1 / np.sqrt(np.diag(Cy_cplx[it,:,:]))))
            part1_cplx = np.arcsin(np.matmul(np.matmul(self.Psi_12, np.real(Cy_cplx)),self.Psi_12))
            part2_cplx = np.arcsin(np.matmul(np.matmul(self.Psi_12, np.imag(Cy_cplx)), self.Psi_12))
            part1 = ut.real2real(part1_cplx)
            part2 = ut.imag2imag(part2_cplx)
            Cr = 2 / np.pi * (part1 + part2)

            #update gmm means from y to r
            A_buss = np.sqrt(2 / np.pi) * self.Psi_12
            # update GMM means
            self.mean_r = np.squeeze(np.matmul(A_buss, np.matmul(A, np.expand_dims(self.means_cplx, axis=2))))
            if self.mean_r.ndim == 1:
                self.gm.means_ = ut.cplx2real(self.mean_r[None, :], axis=1)
            else:
                self.gm.means_ = ut.cplx2real(self.mean_r, axis=1)
            #self.gm.means_ = np.matmul(A_buss, ut.real2cplx(self.gm.means_, ))

            self.gm.covariances_ = Cr.copy()  # this has no effect
            self.gm.precisions_cholesky_ = compute_precision_cholesky(Cr, covariance_type='full')

            # update GMM feature number
            self.gm.n_features_in_ = 2 * A.shape[0]

            r_for_prediction = r
        else:
            raise NotImplementedError(f'Estimation for covariance_type = {self.gm.covariance_type} is not implemented.')

        # precompute the inverse matrices
        Cr_cplx = ut.cov2cov(Cr)
        Cr_inv = np.zeros_like(Cr_cplx)
        for i in range(Cr.shape[0]):
            Cr_inv[i,:,:] = np.linalg.pinv(Cr_cplx[i,:,:])

        return r_for_prediction, Cr_inv


    def estimate_as_sample_from_r(self, r, h_true, snr_dB, n_antennas, A=None, runs_per_sample=1):
        if A is None:
            A = np.eye(n_antennas, dtype=r.dtype)
        r_for_prediction, covs_Cr_inv = self._prepare_for_prediction_1bit(r, A, snr_dB)
        labels = self.gm.predict(ut.cplx2real(r_for_prediction, axis=1))
        h_est = np.zeros([r.shape[0], A.shape[-1]], dtype=complex)
        mse = 0.0
        for yi in range(r.shape[0]):
            mean_h = self.means_cplx[labels[yi], :]
            cov_h = self.covs_cplx[labels[yi], :, :]
            hest_samples = list()
            mses_samples = list()
            for runs in range(runs_per_sample):
                iid_sample = 1 / np.sqrt(2) * (np.random.randn(*h_est.shape[1:],1) +1j*np.random.randn(*h_est.shape[1:],1))
                hest_samples.append(np.squeeze(scilinalg.sqrtm(cov_h) @ iid_sample + np.expand_dims(mean_h, axis=1)))
                mses_samples.append(mse_func(hest_samples[-1], h_true[yi]))
            mse += np.min(mses_samples)
        return mse / r.shape[0]


    def _lmmse_formula_1bit(self, r, mean_h, cov_h, cov_r_inv, mean_r, Psi_12):
        return mean_h + np.sqrt(2 / np.pi) * cov_h @ Psi_12.conj() @ (cov_r_inv @ (r - mean_r))