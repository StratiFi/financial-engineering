import numpy as np
from pykalman import KalmanFilter


class RiskMetrics(object):
    def __init__(self):
        self.alpha_betas = None
        pass

    def get_latest_alpha_beta(self, asset_returns, benchmark_returns):
        if not self.alpha_betas:
            self.build_batch_kalman_estimates(asset_returns, benchmark_returns)
        alpha, beta = self.alpha_betas[-1]
        return alpha, beta

    def build_batch_kalman_estimates(self, asset_returns, benchmark_returns):
        if not len(asset_returns) == len(benchmark_returns):
            raise ('*WTF*')

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
        delta_r = 5e-3
        trans_cov_r = delta_r / (1 - delta_r) * np.eye(2)  # How much random walk wiggles
        list_spy_r = list(benchmark_returns['Adjusted Close'])
        list_vix_r = list(asset_returns['VIX Close'])

        obs_mat_r = np.expand_dims(np.vstack([[list_spy_r], [np.ones(len(benchmark_returns))]]).T, axis=1)

        kf_r = KalmanFilter(n_dim_obs=1, n_dim_state=2,  # y_r is 1-dimensional, (alpha, beta) is 2-dimensional
                            initial_state_mean=[0, 0],
                            initial_state_covariance=np.ones((2, 2)),
                            transition_matrices=np.eye(2),
                            observation_matrices=obs_mat_r,
                            observation_covariance=.01,
                            transition_covariance=trans_cov_r)

        state_means_r, _ = kf_r.filter(asset_returns.values)
        self.alpha_betas = state_means_r

        # # Plot data and use colormap to indicate the date each point corresponds to
        # cm = plt.get_cmap('jet')
        # colors = np.linspace(0.1, 1, len(spy_r))
        # sc = plt.scatter(spy_r, vix_r, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
        # cb = plt.colorbar(sc)
        # cb.ax.set_yticklabels([str(p.date()) for p in spy_r[::len(spy_r)//9].index])
        # plt.xlabel('SPY daily return')
        # plt.ylabel('VIX daily return')


        # Plot every fifth line
        # step = 20
        # xi = np.linspace(-0.5, 0.5, 5)
        # colors_l= np.linspace(0.1, 1, len(state_means_r[::step]))
        # for i, beta in enumerate(state_means_r[::step]):
        #     plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))
        #
        # plt.axis([-0.05,0.05,-0.3, 0.3])
        #
        # # Plot the OLS regression line
        # alpha = np.polyfit(list_spy_r, list_vix_r, 1)[0]
        # beta = np.polyfit(list_spy_r, list_vix_r, 1)[1]
        # plt.plot(xi, poly1d((alpha, beta))(xi), '0.4')
