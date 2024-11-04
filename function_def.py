#minmax归一化
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
def min_max_normal(series):
    return (series-series,min())/(series.max()-series.min())

def normalizing_data(df,seed=42):
    scaler=MinMaxScaler()
    df_normalized=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return df_normalized

def weigths_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias,0)


def plot_gmm(gm, X, label=True, ax=None):
    X = latents
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=200)
    ax = axs or plt.gca()
    labels = gm.fit(X).predict(X)
    if label:
        low_cu = raw_x[:, 5] < 0.05
        low_cu_latent = latents[low_cu]
        low_cu_color = raw_y[:][low_cu]

        high_cu = raw_x[:, 5] >= 0.05
        high_cu_latent = latents[high_cu]
        high_cu_color = raw_y[:][high_cu]

        scatter1 = axs.scatter(low_cu_latent[:, 0], low_cu_latent[:, 1], c=low_cu_color, alpha=.65, s=8, linewidths=0,
                               cmap='viridis')
        scatter2 = axs.scatter(high_cu_latent[:, 0], high_cu_latent[:, 1], c=high_cu_color, alpha=.65, s=14,
                               linewidths=0, cmap='Reds', marker='^')
        # scatter3 = axs.scatter(latents[698:,0], latents[698:,1], alpha=1., s=10, linewidths=.75, edgecolors='k', facecolors='none')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=5, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gm.weights_.max()
    for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
        draw_ellipse(pos, covar, alpha=0.75 * w * w_factor, facecolor='slategrey', zorder=-10)
