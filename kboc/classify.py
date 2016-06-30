import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .scancode import lookup_key, MODIFIER_KEYS


def eer_from_scores(scores):
    """
    Compute the eer from a dataframe with genuine and score columns
    """
    far, tpr, thresholds = roc_curve(scores['genuine'], scores['score'])
    frr = (1 - tpr)
    idx = np.argmin(np.abs(far - frr))
    return np.mean([far[idx], frr[idx]])


def stratified_kfold(df, n_train_samples, n_folds):
    """
    Create stratified k-folds from an indexed dataframe
    """
    sessions = pd.DataFrame.from_records(list(df.index.unique())).groupby(0).apply(lambda x: x[1].unique())
    indexes = set(df.index.unique())
    folds = []
    for i in range(n_folds):
        train_idx = sessions.apply(lambda x: pd.Series(np.random.choice(x, n_train_samples, replace=False)))
        train_idx = pd.DataFrame(train_idx.stack().reset_index(level=1, drop=True)).set_index(0,
                                                                                              append=True).index.values
        test_idx = list(indexes.difference(train_idx))
        folds.append((df.loc[train_idx], df.loc[test_idx]))

    return folds


def events2keynames(events):
    """
    Extract keynames from a sequence of action/scancode events
    """
    keynames = []
    for event in events:
        scancode = event[2:4]
        action = event[4]
        keyname = lookup_key(scancode)

        if action == 'D':
            keynames.append(keyname)
    return keynames


def target_match_keys(target, given):
    """
    Determine the sequence of given keystrokes that closely matches the target sequence
    """
    idx = []

    target = np.array(target)
    given = np.array(given)

    def closest_key_index(seq, i, key):
        matches = np.where(seq == key)[0]
        if len(matches) > 0:
            return matches[np.argmin(np.abs(matches - i))]
        else:
            return i

    # For each target key, find the closest matching key in the given sequence
    # TODO: this could be improved quite a bit
    for i, key in enumerate(target):
        idx.append(closest_key_index(given, i, key))

    return np.array(idx)


def character_keys_only(keynames):
    keys, idx = zip(*[(k, i) for i, k in enumerate(keynames) if k not in MODIFIER_KEYS])
    return list(keys), np.array(list(idx))


def timepressrelease(X, target_keynames, align):
    """
    Extract features given the events, time deltas, and expected key sequence (ie. the name that should be typed)
    """
    keysdown = {}
    time = 0
    timepress, timerelease, keynames = [], [], []

    for event, tdelta in X:
        scancode = event[2:4]
        action = event[4]
        keyname = lookup_key(scancode)

        time += tdelta

        if action == 'D':
            keysdown[keyname] = time
        elif action == 'U' and keyname in keysdown.keys():
            tp = keysdown[keyname]
            timepress.append(tp)
            timerelease.append(time)
            keynames.append(keyname)
            del keysdown[keyname]

    timepress = np.array(timepress)
    timerelease = np.array(timerelease)

    if align == 'keymatch':
        # TODO: these should really be sorted by presstime first, but it doesn't matter much since keymatch does this
        path_idx = target_match_keys(target_keynames, keynames)
        return timepress[path_idx], timerelease[path_idx]
    elif align == 'truncate':
        # Sort by press time and truncate
        path_idx = np.argsort(timepress)
        timepress = timepress[path_idx]
        timerelease = timerelease[path_idx]
        return timepress[:len(target_keynames)], timerelease[:len(target_keynames)]
    elif align == 'drop':
        # Sort by press time, then drop any modifier keys
        path_idx = np.argsort(timepress)
        timepress = timepress[path_idx]
        timerelease = timerelease[path_idx]
        keynames = np.array(keynames)[path_idx]
        _, keys_idx = character_keys_only(keynames)
        timepress = timepress[keys_idx]
        timerelease = timerelease[keys_idx]
        return timepress, timerelease
    else:
        raise Exception('Invalid align method:', align)


def fixedtext_features(X, target_keynames, align):
    """
    Extract fixed-text features (press-press latency and key-hold duration) from the raw data sample X, given the
    target key sequence and keystroke alignment (correspondence) method.
    """
    timepress, timerelease = timepressrelease(X, target_keynames, align)

    f = np.empty(len(target_keynames) * 2 - 1, dtype=np.float32)

    f[0::2][:len(timepress)] = timerelease - timepress
    f[1::2][:(len(timepress) - 1)] = np.diff(timepress)

    return f


def freetext_features(X):
    """
    Extract a sequence of freetext features from the raw data vector X.
    Returns the [press-press latency, duration] observations and keyname events.
    """
    keysdown = {}
    time = 0
    timepress, timerelease, keynames = [], [], []

    for event, tdelta in X:
        scancode = event[2:4]
        action = event[4]
        keyname = lookup_key(scancode)

        time += tdelta

        if action == 'D':
            keysdown[keyname] = time
        elif action == 'U' and keyname in keysdown.keys():
            tp = keysdown[keyname]
            timepress.append(tp)
            timerelease.append(time)
            keynames.append(keyname)
            del keysdown[keyname]

    timepress = np.array(timepress)
    timerelease = np.array(timerelease)
    keynames = np.array(keynames)

    idx = np.argsort(timepress)
    timepress = timepress[idx]
    timerelease = timerelease[idx]
    keynames = keynames[idx]

    obs = np.c_[
        # press-press latency
        np.r_[0, np.diff(timepress)],

        # duration
        timerelease - timepress,
    ]
    for i in range(obs.shape[1]):
        obs[obs[:, i] == 0, i] = np.median(obs[:, i])

    events = keynames
    return obs, events


class ClassifierFixedText(object):
    """
    A fixed-text anomaly detector. The underlying user models expect feature vectors of fixed length.
    """

    def __init__(self, model_factory, feature_normalization='stddev', align='keymatch'):
        self.model_factory = model_factory
        self.feature_normalization = feature_normalization
        self.align = align

        self.models = {}
        self.target_inputs = {}

        self.duration_mins = {}
        self.duration_maxs = {}

        self.latency_mins = {}
        self.latency_maxs = {}
        return

    def fit(self, X, y):
        y = np.array(y)
        unique_y = np.unique(y)
        for yi in unique_y:
            # Neural network models for user T192 originally did not converge due to parameter initialization
            # Use a different seed to choose different initial parameters. The models seem to converge with seed 2016.
            if yi == 'T192':
                np.random.seed(2016)

            Xi = X[y == yi]

            target_input = events2keynames(min(Xi, key=len)[:, 0])

            if self.align == 'drop':
                target_input, _ = character_keys_only(target_input)

            self.target_inputs[yi] = target_input
            Xi = np.array([fixedtext_features(x, self.target_inputs[yi], align=self.align) for x in Xi])
            self.models[yi] = self.model_factory()

            if self.feature_normalization == 'stddev':
                self.duration_mins[yi] = Xi[:, 0::2].mean() - Xi[:, 0::2].std()
                self.duration_maxs[yi] = Xi[:, 0::2].mean() + Xi[:, 0::2].std()
                self.latency_mins[yi] = Xi[:, 1::2].mean() - Xi[:, 1::2].std()
                self.latency_maxs[yi] = Xi[:, 1::2].mean() + Xi[:, 1::2].std()
            elif self.feature_normalization == 'minmax':
                self.duration_mins[yi] = Xi[:, 0::2].min()
                self.duration_maxs[yi] = Xi[:, 0::2].max()
                self.latency_mins[yi] = Xi[:, 1::2].min()
                self.latency_maxs[yi] = Xi[:, 1::2].max()

            Xi = self.normalize(Xi, yi)

            from tensorflow.python.framework import ops
            ops.reset_default_graph()

            self.models[yi].fit(Xi)
        return

    def score(self, X, y):
        scores = np.zeros(len(y))
        for i, (Xi, yi) in enumerate(zip(X, y)):
            Xi = fixedtext_features(Xi, self.target_inputs[yi], align=self.align)
            Xi = self.normalize(Xi[np.newaxis, :], yi).squeeze()
            scores[i] = self.models[yi].score(Xi)
        return scores

    def normalize(self, Xi, yi):
        Xi[:, 0::2] = (Xi[:, 0::2] - self.duration_mins[yi]) / (self.duration_maxs[yi] - self.duration_mins[yi])
        Xi[:, 1::2] = (Xi[:, 1::2] - self.latency_mins[yi]) / (self.latency_maxs[yi] - self.latency_mins[yi])
        Xi[Xi < 0] = 0
        Xi[Xi > 1] = 1
        return Xi


class ClassifierFreeText(object):
    """
    A free-text anomaly detector. The underlying user model operates on sequences of any length.
    """

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.models = {}
        self.inputs = {}
        return

    def fit(self, X, y):
        y = np.array(y)
        unique_y = np.unique(y)
        for yi in unique_y:
            Xi = X[y == yi]
            self.models[yi] = self.model_factory()
            obs, events = zip(*[freetext_features(x) for x in Xi])
            self.models[yi].fit(obs, events)
        return

    def score(self, X, y):
        scores = np.zeros(len(y))
        for i, (Xi, yi) in enumerate(zip(X, y)):
            obs, events = freetext_features(Xi)
            scores[i] = self.models[yi].score(obs, events)
        return scores


def validation(cl, df_genuine, df_impostor, n_genuine_samples=4, n_folds=10, seed=1234, score_normalization='stddev',
               ensemble=False):
    """
    Given a classifier and genuine and impostor sample dataframes, perform Monte Carlo validation
    """
    np.random.seed(seed)

    folds = stratified_kfold(df_genuine, n_genuine_samples, n_folds)

    impostor_idx, impostor_samples = zip(*df_impostor.groupby(level=[0, 1]))
    impostor_labels, _ = zip(*impostor_idx)
    impostor_samples = np.array([x.values for x in impostor_samples])

    results = []
    for i in range(n_folds):
        train, test = folds[i]

        train_idx, train_samples = zip(*train.groupby(level=[0, 1]))
        test_idx, test_samples = zip(*test.groupby(level=[0, 1]))

        train_labels, _ = zip(*train_idx)
        test_labels, _ = zip(*test_idx)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        train_samples = np.array([x.values for x in train_samples])
        test_samples = np.array([x.values for x in test_samples])

        cl.fit(train_samples, train_labels)

        if ensemble:
            genuine_scores = cl.score(i, True)
            impostor_scores = cl.score(i, False)
        else:
            genuine_scores = cl.score(test_samples, test_labels)
            impostor_scores = cl.score(impostor_samples, impostor_labels)

        score_types = np.array([True] * len(genuine_scores) + [False] * len(impostor_scores))
        scores = np.r_[genuine_scores, impostor_scores]
        labels = np.r_[test_labels, impostor_labels]

        if score_normalization is not None:
            for label in np.unique(labels):
                label_idx = labels == label
                if score_normalization == 'minmax':
                    # Normalize scores between min and max for each claimed user (each model)
                    min_score, max_score = scores[label_idx].min(), scores[label_idx].max()
                elif score_normalization == 'stddev':
                    # Normalize scores between +/- 2 std devs of the scores of each model
                    min_score = scores[label_idx].mean() - 2 * scores[label_idx].std()
                    max_score = scores[label_idx].mean() + 2 * scores[label_idx].std()
                else:
                    raise Exception('Unrecognized score normalization:', score_normalization)

                scores[label_idx] = (scores[label_idx] - min_score) / (max_score - min_score)

            scores[scores < 0] = 0
            scores[scores > 1] = 1

        results.extend(zip([i] * len(scores), score_types, scores))

    results = pd.DataFrame(results, columns=['fold', 'genuine', 'score'])

    # Global EER summary over the folds
    summary = results.groupby('fold').apply(eer_from_scores).describe()
    return results, summary


def submission(cl, genuine, unknown, seed=1234, score_normalization='stddev', ensemble=False):
    """
    Given a classifier model and genuine and unknown sample dataframes, generate a submission file.
    """
    np.random.seed(seed)

    genuine_idx, genuine_samples = zip(*genuine.groupby(level=[0, 1]))
    genuine_labels, _ = zip(*genuine_idx)
    genuine_samples = np.array([x.values for x in genuine_samples])

    unknown_idx, unknown_samples = zip(*unknown.groupby(level=[0, 1]))
    unknown_labels, unknown_sessions = zip(*unknown_idx)
    unknown_samples = np.array([x.values for x in unknown_samples])

    unknown_labels = np.array(unknown_labels)
    unknown_sessions = np.array(unknown_sessions)

    cl.fit(genuine_samples, genuine_labels)

    # If this is an ensemble classifier, the score method also needs the unknown sample sessions
    if ensemble:
        scores = cl.score(unknown_samples, unknown_labels, unknown_sessions)
    else:
        scores = cl.score(unknown_samples, unknown_labels)

    if score_normalization is not None:
        for label in np.unique(unknown_labels):
            label_idx = unknown_labels == label
            if score_normalization == 'minmax':
                # Normalize scores between min and max for each claimed user (each model)
                min_score, max_score = scores[label_idx].min(), scores[label_idx].max()
            elif score_normalization == 'stddev':
                # Normalize scores between +/- 2 std devs of the scores of each model
                min_score = scores[label_idx].mean() - 2 * scores[label_idx].std()
                max_score = scores[label_idx].mean() + 2 * scores[label_idx].std()
            else:
                raise Exception('Unrecognized score normalization:', score_normalization)

            scores[label_idx] = (scores[label_idx] - min_score) / (max_score - min_score)

        scores[scores < 0] = 0
        scores[scores > 1] = 1

    scores = pd.DataFrame(scores, columns=['score'],
                          index=pd.MultiIndex.from_tuples(unknown_idx, names=['user', 'session'])).sort_index()

    return scores
