import numpy as np
from pykalman import KalmanFilter


class RiskMetrics(object):
    def __init__(self, period='D'):
        self.alpha_betas_kalman = None
        self.alpha_betas_lstsq = None
        self._alphas = None
        self._betas = None
        self.time_indices = None
        self.period = period
        if self.period == 'D':
            self.lookback = 63
        else:
            self.lookback = 36

    def get_beta_time_series_kalman(self):
        self._betas = [x[0] for x in self.alpha_betas_kalman]
        return self._betas

    def get_alpha_time_series_kalman(self):
        self._alphas = [x[1] for x in self.alpha_betas_kalman]
        return self._alphas

    def get_beta_time_series_lstsq(self):
        self._betas = [x[1] for x in self.alpha_betas_lstsq]
        return self._betas

    def get_alpha_time_series_lstsq(self):
        self._alphas = [x[0] for x in self.alpha_betas_lstsq]
        return self._alphas

    def get_latest_alpha_beta_lstsq(self, asset_returns, benchmark_returns):
        if not self.alpha_betas_lstsq:
            self.build_batch_lstsq_estimates(asset_returns, benchmark_returns)
        alpha, beta = self.alpha_betas_lstsq[-1]
        return alpha, beta

    def get_latest_alpha_beta_kalman(self, asset_returns, benchmark_returns, trans_cov_r=1.e-2):
        if not self.alpha_betas_kalman:
            self.build_batch_kalman_estimates(asset_returns, benchmark_returns, trans_cov_r)
        alpha, beta = self.alpha_betas_kalman[-1]
        return alpha, beta

    def build_batch_kalman_estimates(self, asset_returns, benchmark_returns, trans_cov_r, time_indices=None):
        if not len(asset_returns) == len(benchmark_returns):
            raise '*WTF*'

        #
        # # Plot data and use colormap to indicate the date each point corresponds to
        # cm = plt.get_cmap('jet')
        # colors = np.linspace(0.1, 1, len(spy_r))
        # sc = plt.scatter(spy_r, vix_r, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
        # cb = plt.colorbar(sc)
        # cb.ax.set_yticklabels([str(p.date()) for p in spy_r[::len(spy_r)//9].index])
        # plt.xlabel('SPY daily return')
        # plt.ylabel('VIX daily return')

        # Run Kalman filter on returns data
        delta_r = trans_cov_r
        trans_cov_r = delta_r / (1 - delta_r) * np.eye(2)  # How much random walk wiggles
        # list_spy_r = list(benchmark_returns['Adjusted Close'])
        # list_vix_r = list(asset_returns['VIX Close'])

        obs_mat_r = np.expand_dims(np.vstack([[benchmark_returns], [np.ones(len(benchmark_returns))]]).T, axis=1)

        initial_state_mean = [0, 0]

        # if we have enough statistics we want to set the initial values using some LLS estimate
        len_r = len(benchmark_returns)
        if len_r >= 15:
            initial_state_mean[0], initial_state_mean[1] = np.polyfit(benchmark_returns[0:min(len_r, 30)],
                                                 asset_returns[0:min(len_r, 30)], 1)

        # See Chap. 3 of Sarkka, using same conventions
        kf_r = KalmanFilter(n_dim_obs=1, n_dim_state=2,  # y_r is 1-dimensional, (alpha, beta) is 2-dimensional
                            initial_state_mean=initial_state_mean,
                            initial_state_covariance=2. * np.ones((2, 2)),
                            transition_matrices=np.eye(2),
                            observation_matrices=obs_mat_r,
                            observation_covariance=.005,
                            transition_covariance=trans_cov_r)

        if time_indices is not None:
            self.time_indices = time_indices

        state_means_r, _ = kf_r.filter(asset_returns)
        self.alpha_betas_kalman = state_means_r

    def build_batch_lstsq_estimates(self, asset_returns, benchmark_returns):
        if not len(asset_returns) == len(benchmark_returns):
            raise '*WTF*'

        # Run Kalman filter on returns data
        beta = np.zeros(len(asset_returns))
        alpha = np.zeros(len(asset_returns))
        for enum_i, elem in enumerate(asset_returns):
            lookback = min(self.lookback, enum_i)
            # print '==> ', enum_i, len(asset_returns), len(beta)
            beta[enum_i], alpha[enum_i] = np.polyfit(benchmark_returns[enum_i - lookback:enum_i + 1],
                                                     asset_returns[enum_i - lookback:enum_i + 1], 1)

        # don't wanna do a line fit for less than 3 points, really
        beta[0], alpha[0] = 0, 0
        beta[1], alpha[1] = 0, 0

        self.alpha_betas_lstsq = np.array(zip(alpha, beta))
