import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

def plot_pred(total_trues, total_preds, pred_idx=0, col_idx=0, span=300):
    fig = plt.figure()
    lst1, lst2 = [], []
    for i in range(total_trues.shape[0]):
        lst1.append(total_trues[i][pred_idx,col_idx])
        lst2.append(total_preds[i][pred_idx,col_idx])
    lst1 = np.array(lst1)
    lst2 = np.array(lst2)
    span = min(total_trues.shape[0], span)
    start_idx = np.random.choice(total_trues.shape[0]-span+1, 1)[0]
    plt.plot(lst1[start_idx:start_idx+span])
    plt.plot(lst2[start_idx:start_idx+span])
    # plt.plot(lst1[start_idx:start_idx+span], "o")
    # plt.plot(lst2[start_idx:start_idx+span], "o")
    return fig

def plot_errorbar(y, ax, percentile=85, **kwargs):
    mean = y.mean(axis=0)
    yerr = np.stack([
        mean - np.percentile(y, 100-percentile, axis=0),
        np.percentile(y, percentile, axis=0) - mean
    ])

    ax.errorbar(np.arange(mean.shape[0]), mean, yerr=yerr, **kwargs)


def plot_errors_threshold(y_true, y_pred, ax, error_band=0.1, unit='', **kwargs):
    threshold_initial = max(
        np.sort(np.abs(y_true-y_pred), axis=0)[int(y_true.shape[0]*(1 - error_band))])
    threshold_range = np.linspace(
        threshold_initial*0.75, threshold_initial*1.25, 5)

    for threshold in threshold_range:
        number_mispredictions = np.where(
            np.abs(y_true-y_pred) > threshold, 1, 0).sum(axis=0) / y_true.shape[0] * 100

        if threshold == threshold_initial:
            ax.plot(number_mispredictions, '-',
                    label=f'threshold {threshold:.1f}{unit}', linewidth=5)
        else:
            ax.plot(number_mispredictions, '-',
                    label=f'threshold {threshold:.1f}{unit}')

    ax.axhline(y=error_band*100, color='black')
    ax.text(25, error_band*100+0.5, f'{int(error_band*100)}%')

    ax.set_xlabel('timesteps')
    ax.set_ylabel('% mispredicted')
    ax.legend()

    return threshold_initial

def plot_dataset_distribution(x, ax, alpha=0.4, unit='', **kwargs):
    plot_errorbar(y=x, ax=ax, color='black', ls='dotted',
                marker='s', markeredgecolor='black', markerfacecolor='white', markeredgewidth=2, linewidth=2)

    sns.stripplot(data=x, linewidth=0, color='b',
                alpha=alpha, zorder=1, marker='.', s=8, ax=ax)

    ax.set_xlabel('timesteps')
    ax.legend()


def plot_values_distribution(y_true, y_pred, ax, alpha=0.4, unit='', **kwargs):
    plot_errorbar(y=y_true, ax=ax, color='darkgreen', ls='dotted',
                  marker='s', markeredgecolor='darkgreen', markerfacecolor='white', markeredgewidth=2, linewidth=2, label='true mean')

    plot_errorbar(y=y_pred, ax=ax, color='black', ls='dotted',
                  marker='s', markeredgecolor='black', markerfacecolor='white', markeredgewidth=2, linewidth=2, label='prediction mean')

    sns.stripplot(data=y_true, linewidth=0, color='g',
                  alpha=alpha, zorder=1, marker='.', s=8, ax=ax)

    sns.stripplot(data=y_pred, linewidth=0, color='b',
                  alpha=alpha, zorder=1, marker='.', s=8, ax=ax)

    ax.set_xlabel('timesteps')
    ax.set_ylabel(unit)
    ax.legend()


def plot_error_distribution(y_true, y_pred, ax, alpha=0.4, unit='', **kwargs):
    diff = y_true-y_pred
    plot_errorbar(y=diff, ax=ax, color='black', ls='dotted',
                  marker='s', markeredgecolor='black', markerfacecolor='white', markeredgewidth=2, linewidth=2, label='mean')
    sns.stripplot(data=diff, linewidth=0, color='orange',
                  alpha=alpha, zorder=1, marker='.', s=8, ax=ax)

    ax.set_xlabel('timesteps')
    ax.set_ylabel(unit)
    ax.legend()


def plot_visual_sample(y_true, y_pred, ax, unit='', **kwargs):
    # Select training example
    idx = np.random.randint(0, y_true.shape[0])
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    # Add title, axis and legend
    ax.plot(y_true, label="Truth")
    ax.plot(y_pred, label="Prediction")

    ax.set_xlabel('timesteps')
    ax.set_ylabel(unit)
    ax.legend()
