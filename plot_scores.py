import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    classifier = sys.argv[1]
    df_validation = pd.read_csv('validations/' + classifier + '.csv', index_col=[0, 1, 2])['score']
    df_submission = pd.read_csv('submissions/' + classifier + '.txt', index_col=0, sep=' ', header=None)[1]

    BINS = np.linspace(df_submission.min(), df_submission.max(), 31)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 4))

    split_validation = True
    if split_validation:
        sns.distplot(df_validation.xs(True, level='genuine').values, ax=ax[0], norm_hist=True, color='green',
                     kde_kws={'label': 'Genuine'}, bins=BINS)
        sns.distplot(df_validation.xs(False, level='genuine').values, ax=ax[0], norm_hist=True, color='red',
                     kde_kws={'label': 'Impostor'}, bins=BINS)
        plt.legend(loc='upper left')
    else:
        sns.distplot(df_validation.values, ax=ax[0], norm_hist=True, bins=BINS)

    ax[0].text(0.5, 0.95, 'Validation',
               horizontalalignment='center',
               verticalalignment='top',
               transform=ax[0].transAxes)
    ax[0].set_ylabel('Density')

    sns.distplot(df_submission.values, ax=ax[1], norm_hist=True, kde_kws={'label': 'Unknown'}, bins=BINS)
    plt.legend(loc='upper left')
    ax[1].text(0.5, 0.95, 'Test',
               horizontalalignment='center',
               verticalalignment='top',
               transform=ax[1].transAxes)
    ax[1].set_ylabel('Density')
    ax[1].set_xlabel('Score')

    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], bbox_inches='tight')
    else:
        plt.show()
