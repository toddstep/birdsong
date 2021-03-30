import librosa
import numpy as np
from sklearn import base, mixture


def iterate(X, num_initial_clusters):
    """
    Iteratively perform agglomerative clustering.
    The iterating continues until no model pairs meet the merging criteria.
    The models and associated segmentation is returned.
    
    Keyword arguments:
    X -- sequence of features
    num_initial_clusters -- number of models to initialize
    segmentation -- Labels for each item in sequence
    """

    models = init_models(X, num_initial_clusters, num_components=1)
    for i in range(num_initial_clusters-1):
        good_segmentation = False
        while not good_segmentation:
            states = decode_states(X, models)
            pruned_models = prune(models, states)
            if len(pruned_models) == len(models):
                good_segmentation = True
            else:
                models = pruned_models
        models = fit_models(X, models, states)
        models, did_merge = merge(X, models, states)
        if not did_merge:
            break
    return models, states

def init_models(X, num_models, num_components=4, covariance_type='diag'):
    """
    Compute initial models.
    
    Keyword arguments:
    X -- Sequence of features
    num_models -- Number of initial GMMs
    num_components -- Number of mixtures in each GMM (default 4)
    covariance_type -- Covariance type for each components (default 'diag')
    """
    
    num_frames = len(X)
    segmentation = np.linspace(0, num_models, num_frames, endpoint=False).astype(int)
    basic_model = mixture.GaussianMixture(num_components, covariance_type, reg_covar=1e-05,
                                          random_state=0)
    models = []
    for i in range(num_models):
        m = base.clone(basic_model).fit(X[segmentation==i])
        models.append(m)
    return models
    
def fit_models(X, models, segmentation):
    """
    Reestimate Gaussian Mixture Models (GMMs).
        
    Keyword arguments:
    X -- Sequence of features
    models -- Set of models    
    segmentation -- Labels for each item in sequence
    """
    
    models = models.copy()
    for i, m in enumerate(models):
        X_segments = X[segmentation==i]
        n_frames = len(X_segments)
        n_components = m.get_params()['n_components']
        if n_frames < n_components:
            print(f'Too few frames for model {i}. '
                  f'Reducing number of components from {n_components} to {n_frames}.')
            m.set_params(n_components=n_frames)
        m.fit(X_segments)
    return models

def transition_matrix(num_models, min_duration, self_trans=0.6):
    """
    Create a transition matrix that imposes a minimum state duration.
    There will be `num_models` * `min_duration` rows (i.e., states) in the square matrix.
    Each model will be represented by `min_duration` states in the matrix.
    For the last state of a model, there will be `num_models` + 1 allowed transitions:
        - a self-transition
        - transitions to the initial state of each of the `num_models` models
    For the other states of a model, there will be only two allowed transitions:
        - a self-transition
        - a transition to the next state of the model
    
    For example:
    transition_matrix(2, 3):
    array([[0.6, 0.4, 0. , 0. , 0. , 0. ],
           [0. , 0.6, 0.4, 0. , 0. , 0. ],
           [0.2, 0. , 0.6, 0.2, 0. , 0. ],
           [0. , 0. , 0. , 0.6, 0.4, 0. ],
           [0. , 0. , 0. , 0. , 0.6, 0.4],
           [0.2, 0. , 0. , 0.2, 0. , 0.6]])

    Keyword arguments:
    num_models -- Number of GMMs
    min_duration -- Minimum number of frames to remain in a GMM
    self_trans -- Probability for self-transition
    """
    
    state_step = min_duration
    total_states = num_models * state_step
    trans = np.zeros((total_states, total_states))
    self_trans_comp = 1 - self_trans
    for state in range(0, total_states):
        trans[state, state] = self_trans
        is_last_state = ((state+1) % state_step) == 0
        if is_last_state:
            for initial_state in range(0, total_states, state_step):
                trans[state, initial_state] = self_trans_comp / num_models          
        else:
            trans[state, state+1] = self_trans_comp
    return trans

def min_duration_like_matrix(like_matrix, min_duration):
    """
    Prepare a likelihood matrix to be used for minimum duration Viterbi decoding.
    As the `transition_matrix()` creates multiple rows (& columns) for each model,
    this method also creates multiple rows for each model/probability.
    
    For example:
    min_duration_like_matrix(np.array([[.3, .7],[.4, .6]]), 2)
    array([[0.3, 0.7],
           [0.3, 0.7],
           [0.4, 0.6],
           [0.4, 0.6]])
    
    
    Keyword arguments:
    like_matrix -- likelihood matrix
    min_duration -- Minimum number of frames to remain in a GMM
    """
    return np.repeat(like_matrix, min_duration, 0)

def decode_states(X, models, min_duration=5):
    """
    Do Viterbi decoding of sequence using provided models.
    Assumes every state can transition to another and that self-transitions have a high probability.
    
    Keyword arguments:
    X -- Sequence of features
    models -- Set of models
    min_duration -- Minimum number of frames to remain in a GMM (default 5)
    """
    
    trans = transition_matrix(len(models), min_duration)
    log_like = list(map(lambda m: m.score_samples(X), models))
    like = np.exp(np.array(log_like))
    like = min_duration_like_matrix(like, min_duration)
    # librosa viterbi does not allow probs/likelihoods greater than 1.
    # So, divide by the max likelihood
    states = librosa.sequence.viterbi(like / like.max(), trans)
    model_ids = states // min_duration

    return model_ids

def prune(models, segmentation, thresh=1):
    """
    Remove models that appear at most `thresh` times in segmentation.
    
    Keyword arguments:
    models -- Set of models
    segmentation -- Labels for each item in sequence
    thresh -- Frame count for models that are to be pruned (default 1)
    """
    
    models = models.copy()
    for i in reversed(range(len(models))):
        n_frames = np.count_nonzero(segmentation==i)
        if n_frames <= thresh:
            print(f'Found {n_frames} frames for model {i}.  Pruning. . . .')
            models.pop(i)
    return models
            
    
def merge(X, models, segmentation):
    """
    Merge the most similar models using method of Ajmera (2004).
    This does an exhaustive search over all model pairs.
    A pair is a candidate to merge if, according to the provided segmentation,
    the likelihood increases with a merged model.
    Only the single pair that produces the greatest increase is merged.
    If there is no pair that increases the likelihood, no merging is done;
    in such a case, the same models are returned and did_merge is set to False.
    
    Keyword arguments:
    X -- sequence of features
    models -- Set of models
    segmentation -- Labels for each item in sequence
    """
    
    models = models.copy()
    num_models = len(models)
    highest_diff = None
    best_pair = None
    best_model = None
    did_merge = False
    for m1 in range(num_models-1):
        X1 = X[segmentation==m1]
        score1 = models[m1].score_samples(X1).sum()
        n_components1 = models[m1].get_params()['n_components']
        for m2 in range(m1+1, num_models):
            X2 = X[segmentation==m2]
            score2 = models[m2].score_samples(X2).sum()
            n_components2 = models[m2].get_params()['n_components']
            merged_X = X[np.any([segmentation==m1, segmentation==m2], axis=0)]
            merged_model = base.clone(models[m1]).set_params(n_components=(n_components1+n_components2))
            merged_model.fit(merged_X)
            merged_score = merged_model.score_samples(merged_X).sum()
            pair_diff = merged_score - score1 - score2
            if pair_diff > 0:
                if highest_diff is None or pair_diff > highest_diff:
                    highest_diff = pair_diff
                    best_pair = (m1, m2)
                    best_model = merged_model
    if best_pair is not None:
        print(f'best_pair: {best_pair}')
        models[best_pair[0]] = best_model
        did_merge = True
        models.pop(best_pair[1])

    return models, did_merge

