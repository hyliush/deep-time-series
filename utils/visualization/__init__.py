import numpy as np
from matplotlib import pyplot as plt

from .plot_functions import plot_pred, plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample, plot_dataset_distribution


def map_plot_function_input(dataset, plot_function, plot_kwargs={}, dataset_indices=None, labels=None, limit=None):

    labels = labels or dataset.labels['Z']
    limit = limit or dataset._y.shape[1]
    if dataset_indices is not None:
        dataset_x = dataset._x[dataset_indices].numpy()
    else:
        dataset_x = dataset._x.numpy()

    # Create subplots
    fig, axes = plt.subplots(len(labels), 1)
    fig.set_figwidth(25)
    fig.set_figheight(5*len(labels))
    plt.subplots_adjust(bottom=0.05)

    # Fix for single label
    if len(labels) == 1:
        axes = [axes]

    for label, ax in zip(labels, axes):
        # Get label index from dataset
        idx_label = dataset.labels['Z'].index(label)

        # Select data for time period and label
        x = dataset_x[:, :limit, idx_label]

        plot_function(x, ax, **plot_kwargs)

        ax.set_title(label)

        n_ticks = limit // 24
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % n_ticks:
                label.set_visible(False)


def map_plot_function(trues, preds, plot_function, labels=None, idx_labels=None, limit=None, dataset=None, plot_kwargs={}):

    labels = dataset.labels if dataset else labels
    idx_labels = [dataset.labels.index(label) for label in labels] if dataset else idx_labels
    limit = dataset.pred_len if dataset else limit
    
    # Create subplots
    fig, axes = plt.subplots(len(labels), 1)
    fig.set_figwidth(25)
    fig.set_figheight(5*len(labels))
    plt.subplots_adjust(bottom=0.05)

    # Fix for single label
    if len(labels) == 1:
        axes = [axes]

    for idx_label, label, ax in zip(idx_labels, labels, axes):

        # Select data for time period and label
        y_pred = preds[:, :limit, idx_label]
        y_true = trues[:, :limit, idx_label]

        # If a consumption
        if label.startswith('Q_'):
            unit = 'kW'
        else:
            unit = '°C'

        plot_function(y_true, y_pred, ax, **plot_kwargs, unit=unit)
        ax.set_title(label)

        n_ticks = max(1, limit // 24)
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % n_ticks:
                label.set_visible(False)
