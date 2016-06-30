import re
import os
import sys
import zipfile
import pandas as pd

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

from pohmm import Pohmm
from kboc.anomaly import *
from kboc.classify import *

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Where to save the validation and submission score files
VALIDATION_DIR = os.path.join(ROOT_DIR, 'validations')
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submissions')

# TODO: Place the KBOC data archives in this folder.
# They will be extracted the first time this script is run
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Keystroke database locations (not including extensions)
DEV_GENUINE = os.path.join(DATA_DIR, 'dev_genuine.csv')
DEV_IMPOSTOR = os.path.join(DATA_DIR, 'dev_impostor.csv')
GENUINE = os.path.join(DATA_DIR, 'genuine.csv')
UNKNOWN = os.path.join(DATA_DIR, 'unknown.csv')

# Regex to match the genuine/impostor/unknown files and extract the user/session
# Genuine samples are 1-14, impostor samples are 15-24
DEV_GENUINE_RE = re.compile('(D[0-9]{3})_(0[1-9]|1[0-4]).txt$')
DEV_IMPOSTOR_RE = re.compile('(D[0-9]{3})_(1[5-9]|2[0-4]).txt$')

# Genuine samples are 1-4, unknown samples are 5-24
TEST_GENUINE_RE = re.compile('(T[0-9]{3})_(0[1-4]).txt$')
TEST_UNKNOWN_RE = re.compile('(T[0-9]{3})_(0[5-9]|1[0-9]|2[0-4]).txt$')


def find_file(dirpath, inname):
    """
    Find a file that contains inname (case insensitive)
    """
    fname = [fname for fname in os.listdir(dirpath) if inname in fname.lower()]

    if len(fname) == 0:
        return None
    else:
        return fname[0]


def zipfiles2dataframe(zip, rexpr):
    """
    Extract files in the zip that match the regular expression rexpr and load them as a data frame
    """
    dfs = []
    for f in zip.filelist:
        m = rexpr.search(f.filename)
        if not m:
            continue

        df = pd.read_fwf(StringIO(zip.read(f).decode('utf-8')), header=None, skiprows=1)
        df.columns = ['event', 'tau']
        df['user'], df['session'] = m.groups()
        dfs.append(df)

    df = pd.concat(dfs).set_index(['user', 'session'])

    return df


def extract_dev_db():
    """
    Extracts the development database (genuine and impostor samples), saving as csv files.
    """
    if os.path.exists(DEV_GENUINE) and os.path.exists(DEV_IMPOSTOR):
        # The csv files already exist
        return

    # Look for the zip file in the data dir
    fname = find_file(DATA_DIR, 'development')

    if fname is None:
        raise Exception('Place Development_kit.zip in the data directory:', DATA_DIR)

    zip = zipfile.ZipFile(os.path.join(DATA_DIR, fname))

    df_genuine = zipfiles2dataframe(zip, DEV_GENUINE_RE)
    df_impostor = zipfiles2dataframe(zip, DEV_IMPOSTOR_RE)

    df_genuine.sort_index().to_csv(DEV_GENUINE)
    df_impostor.sort_index().to_csv(DEV_IMPOSTOR)


def extract_test_db():
    """
    Extracts the test database (genuine and unknown samples), saving as csv files.
    """
    if os.path.exists(GENUINE) and os.path.exists(UNKNOWN):
        # The csv files already exist
        return

    # Look for the zip file in the data dir
    fname = find_file(DATA_DIR, 'test')

    if fname is None:
        raise Exception('Place Test_kit.zip in the data directory:', DATA_DIR)

    zip = zipfile.ZipFile(os.path.join(DATA_DIR, fname))

    df_genuine = zipfiles2dataframe(zip, TEST_GENUINE_RE)
    df_unknown = zipfiles2dataframe(zip, TEST_UNKNOWN_RE)

    df_genuine.sort_index().to_csv(GENUINE)
    df_unknown.sort_index().to_csv(UNKNOWN)


def load_db(db):
    """
    Load database as a dataframe. Extracts the zip files if necessary. The database is indexed by the user, session.
    """
    if DEV_GENUINE == db or DEV_IMPOSTOR == db:
        extract_dev_db()

    if GENUINE == db or UNKNOWN == db:
        extract_test_db()

    return pd.read_csv(db, index_col=[0, 1])


def save_validation(df, name):
    """
    Format (4 columns):
    ,fold,genuine,score
    """
    df.to_csv(os.path.join(VALIDATION_DIR, name + '.csv'))
    return


def save_submission(df, name):
    """
    Format:
        T001_01 0.213
        T001_02 0.465
        ...
    """
    df = df.reset_index()
    df['file'] = df['user'] + '_' + df['session'].map(lambda x: '%02d' % x)
    df = df[['file', 'score']].set_index('file')
    df.to_csv(os.path.join(SUBMISSION_DIR, name + '.txt'), sep=' ', header=None, float_format='%.6f')
    return


def validator(cl, name, **kwargs):
    """
    Helper function to validate a model generate a validation score file. Extra args are passed to validation()
    """
    df_dev_genuine = load_db(DEV_GENUINE)
    df_dev_impostor = load_db(DEV_IMPOSTOR)
    classifier_scores, classifier_summary = validation(cl, df_dev_genuine, df_dev_impostor, n_folds=10, **kwargs)
    print('%s EER: %.4f +/- %.4f' % (name, classifier_summary['mean'], classifier_summary['std']))
    save_validation(classifier_scores, name)
    return


def submitter(cl, name, **kwargs):
    """
    Helper function to generate a named submission file given the classifier. Extra args are passed to submission()
    """
    df_genuine = load_db(GENUINE)
    df_unknown = load_db(UNKNOWN)
    cl_submission = submission(cl, df_genuine, df_unknown, **kwargs)
    save_submission(cl_submission, name)
    return


if __name__ == '__main__':
    system1 = ClassifierFixedText(lambda: Autoencoder([5, 4, 3]))
    validator(system1, 'system1')
    submitter(system1, 'system1')

    system2 = ClassifierFixedText(
        lambda: VariationalAutoencoder(dict(n_hidden_recog_1=5,  # 1st layer encoder neurons
                                            n_hidden_recog_2=5,  # 2nd layer encoder neurons
                                            n_hidden_gener_1=5,  # 1st layer decoder neurons
                                            n_hidden_gener_2=5,  # 2nd layer decoder neurons
                                            n_z=3),  # dimensionality of latent space
                                       batch_size=2))
    validator(system2, 'system2')
    submitter(system2, 'system2')

    system3 = ClassifierFreeText(lambda: Pohmm(n_hidden_states=2,
                                               init_spread=2,
                                               emissions=['lognormal', 'lognormal'],
                                               smoothing='freq',
                                               init_method='obs',
                                               thresh=1e-2))
    validator(system3, 'system3')
    submitter(system3, 'system3')

    system4 = ClassifierFixedText(lambda: OneClassSVM())
    validator(system4, 'system4')
    submitter(system4, 'system4')

    system5 = ClassifierFixedText(lambda: ContractiveAutoencoder(400, lam=1.5))
    validator(system5, 'system5')
    submitter(system5, 'system5')

    system6 = ClassifierFixedText(lambda: Manhattan())
    validator(system6, 'system6')
    submitter(system6, 'system6')

    system7 = ClassifierFixedText(lambda: Autoencoder([5]))
    validator(system7, 'system7')
    submitter(system7, 'system7')

    system8 = ClassifierFixedText(lambda: ContractiveAutoencoder(200, lam=0.5))
    validator(system8, 'system8')
    submitter(system8, 'system8')

    top3 = [3, 4, 5]
    validator(MeanEnsemble_csv([os.path.join(VALIDATION_DIR, 'system%d.csv' % i) for i in top3]), 'system9', ensemble=True)
    submitter(MeanEnsemble_txt([os.path.join(SUBMISSION_DIR, 'system%d.txt' % i) for i in top3]), 'system9', ensemble=True)

    validator(MeanEnsemble_csv([os.path.join(VALIDATION_DIR, 'system%d.csv' % i) for i in range(1, 8)]), 'system10',
              ensemble=True)
    submitter(MeanEnsemble_txt([os.path.join(SUBMISSION_DIR, 'system%d.txt' % i) for i in range(1, 8)]), 'system10',
              ensemble=True)

    system11 = ClassifierFreeText(lambda: Pohmm(n_hidden_states=2,
                                                init_spread=2,
                                                emissions=['lognormal', 'lognormal'],
                                                smoothing='freq',
                                                init_method='obs',
                                                thresh=1e-2))
    validator(system11, 'system11', score_normalization='minmax')
    submitter(system11, 'system11', score_normalization='minmax')

    system12 = ClassifierFixedText(lambda: OneClassSVM())
    validator(system12, 'system12', score_normalization='minmax')
    submitter(system12, 'system12', score_normalization='minmax')

    system13 = ClassifierFixedText(lambda: ContractiveAutoencoder(400, lam=1.5))
    validator(system13, 'system13', score_normalization='minmax')
    submitter(system13, 'system13', score_normalization='minmax')

    system14 = ClassifierFixedText(lambda: ContractiveAutoencoder(200, lam=0.5))
    validator(system14, 'system14', score_normalization='minmax')
    submitter(system14, 'system14', score_normalization='minmax')

    validator(MeanEnsemble_csv([os.path.join(VALIDATION_DIR, 'system%d.csv' % i) for i in [11, 12, 13, 14]]),
              'system15', ensemble=True)
    submitter(MeanEnsemble_txt([os.path.join(SUBMISSION_DIR, 'system%d.txt' % i) for i in [11, 12, 13, 14]]),
              'system15', ensemble=True)

    # Manhattan distance with min/max normalization, not submitted
    system16 = ClassifierFixedText(lambda: Manhattan())
    validator(system16, 'system16', score_normalization='minmax')
    submitter(system16, 'system16', score_normalization='minmax')

    # Manhattan distance without normalization, not submitted
    system17 = ClassifierFixedText(lambda: Manhattan())
    validator(system17, 'system17', score_normalization=None)
    submitter(system17, 'system17', score_normalization=None)

    system18 = ClassifierFixedText(lambda: Manhattan(), align='truncate')
    validator(system18, 'system18', score_normalization='stddev')
    submitter(system18, 'system18', score_normalization='stddev')

    # Manhattan distance with min/max normalization, not submitted
    system19 = ClassifierFixedText(lambda: Manhattan(), align='truncate')
    validator(system19, 'system19', score_normalization='minmax')
    submitter(system19, 'system19', score_normalization='minmax')

    # Manhattan distance without normalization, not submitted
    system20 = ClassifierFixedText(lambda: Manhattan(), align='truncate')
    validator(system20, 'system20', score_normalization=None)
    submitter(system20, 'system20', score_normalization=None)

    system21 = ClassifierFixedText(lambda: Manhattan(), feature_normalization='minmax')
    validator(system21, 'system21', score_normalization='stddev')
    submitter(system21, 'system21', score_normalization='stddev')

    system22 = ClassifierFixedText(lambda: Manhattan(), align='drop')
    validator(system22, 'system22', score_normalization='stddev')
    submitter(system22, 'system22', score_normalization='stddev')

    # Manhattan distance with min/max normalization, not submitted
    system23 = ClassifierFixedText(lambda: Manhattan(), align='drop')
    validator(system23, 'system23', score_normalization='minmax')
    submitter(system23, 'system23', score_normalization='minmax')

    # Manhattan distance without normalization, not submitted
    system24 = ClassifierFixedText(lambda: Manhattan(), align='drop')
    validator(system24, 'system24', score_normalization=None)
    submitter(system24, 'system24', score_normalization=None)
