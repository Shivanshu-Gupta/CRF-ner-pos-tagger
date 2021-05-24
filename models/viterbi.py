import numpy as np
from pdb import set_trace

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    try:
        dp = np.zeros([N, L])
        bp = np.zeros([N-1, L], dtype=np.int)
        dp[0] = start_scores + emission_scores[0]
        for i in range(1, N):
            dp[i] = emission_scores[i]
            for j in range(L):
                _j = np.argmax(trans_scores[:, j] + dp[i-1])
                dp[i][j] = emission_scores[i][j] + trans_scores[_j, j] + dp[i - 1][_j]
                bp[i-1][j] = _j
        final_scores = end_scores + dp[N-1]
        best_score = final_scores.max()
        best_seq = [np.argmax(final_scores)]
        for i in range(N-1):
            best_seq.append(bp[N-2-i][best_seq[-1]])
        best_seq.reverse()
    except:
        set_trace()
    return (best_score, best_seq)
    # y = []
    # for i in range(N):
    #     # stupid sequence
    #     y.append(i % L)
    # score set to 0
    # return (0.0, y)
