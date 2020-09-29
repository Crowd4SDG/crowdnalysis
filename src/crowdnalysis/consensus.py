import numpy as np

def compute_counts(m, I, J):
    n = np.zeros((I, J))
    for i, k, j in m:
        n[i, j] += 1
    return n
    # print(n)

def majority_voting(m, I, J):
    n = compute_counts(m, I, J)
    # print(n)
    best_count = np.amax(n, axis=1)
    num_best_candidates = np.sum((n==best_count[:, np.newaxis]), axis=1)
    best = np.argmax(n, axis=1)
    best[num_best_candidates != 1] = -1
    return best

def get_question_matrix_and_ranges(d, question):
    m = d.get_question_matrix(question)
    I = d.n_tasks
    J = d.n_labels(question)
    return m, I, J

def compute_majority_voting(d, question):
    m, I, J = get_question_matrix_and_ranges(d, question)
    return majority_voting(m, I, J)

def probabilistic_consensus(m, I, J, softening=0.1):
    n = compute_counts(m, I, J)
    n += softening
    return n / np.sum(n, axis=1)[:, np.newaxis]

def compute_probabilistic_consensus(d, question, softening=0.1):
    m, I, J = get_question_matrix_and_ranges(d, question)
    return probabilistic_consensus(m, I, J, softening)

class Dawid_Skene:
    def __init__(self):
        pass

    def compute_consensus(self, d, question, max_iterations=10000, tolerance=1e-7, prior=0.0):
        m = d.get_question_matrix(question)
        self.I = d.n_tasks
        self.J = d.n_labels(question)
        self.K = d.n_annotators

        self.n = self.compute_n(m)
        # First estimate of T_{i,j} is done by probabilistic consensus
        self.T = compute_probabilistic_consensus(d, question, softening=prior)

        # Initialize the percentages of each label
        self.p = self.m_step_p(self.T, prior)

        #print("p=", self.p)

        # Initialize the errors
        # _pi[k,j,l] (KxJxJ)
        self.logpi = self.m_step_logpi(self.T, self.n, prior)
        #self.logpi = np.log(self.pi)

        #print("pi=", np.exp(self.logpi))

        has_converged = False
        num_iterations = 0
        while num_iterations < max_iterations and not has_converged:
            # Expectation step
            old_T = self.T
            self.T = self.e_step(self.n, self.logpi, self.p)
            # Maximization
            self.p = self.m_step_p(self.T, prior)
            #print("p=", self.p)
            self.logpi = self.m_step_logpi(self.T, self.n, prior)
            #print("pi=", np.exp(self.logpi))
            has_converged = np.allclose(old_T, self.T, atol=tolerance)
            num_iterations += 1
        if has_converged:
            print("DS has converged in", num_iterations, "iterations")
        else:
            print("The maximum of", max_iterations, "iterations has been reached")
        return self.p, np.exp(self.logpi), self.T

    def compute_n(self, m):

        # print(m)
        #N = m.shape[0]

        # print("N=", N, "I=", self.I, "J=", self.J, "K=", self.K)

        # Compute the n matrix

        n = np.zeros((self.K, self.I, self.J))
        for i, k, j in m:
            n[k, i, j] += 1
        return n

    def m_step_p(self, T, prior):
        p = np.sum(T, axis=0)
        p += prior
        p /= np.sum(p)
        return p

    def m_step_logpi(self, T, n, prior):
        _pi = np.swapaxes(np.dot(T.transpose(), n), 0, 1)
        _pi += prior
        sums = np.sum(_pi, axis=2)
        _pi /= sums[:, :, np.newaxis]
        return np.log(_pi)

    def e_step(self, n, logpi, p):
        T = np.exp(np.tensordot(n, logpi, axes=([0, 2], [0, 2])))  # IxJ
        T *= p[np.newaxis, :]
        T /= np.sum(T, axis=1)[:, np.newaxis]
            # Potential numerical error here.
            # Plan for using smthg similar to the logsumexp trick in the future
        return T

    def compute_crossed_error(self, d, question, T, prior=0.0):
        m = d.get_question_matrix(question)
        self.I = d.n_tasks
        self.J = d.n_labels(question)
        self.K = d.n_annotators
        n = self.compute_n(m)
        return np.exp(self.m_step_logpi(T, n, prior))

    def sample(self, p, _pi, I, num_annotators):
        # TODO: Consider using pyAgrum
        # _pi = np.exp(logpi)
        J = len(p)
        K = _pi.shape[0]
        # Sample the real labels
        real_labels = np.random.choice(J, size=I, p=p)
        # Sample the annotators
        annotators = np.random.choice(K, size=(I, num_annotators))
        crowd_labels = np.zeros((I * num_annotators, 3), dtype=np.int32)

        for i in range(I):
            for i_a in range(num_annotators):
                annotator_index = annotators[i, i_a]
                task_run_index = i * num_annotators + i_a
                crowd_labels[task_run_index, 0] = i
                crowd_labels[task_run_index, 1] = annotator_index
                crowd_labels[task_run_index, 2] = np.random.choice(J, size=1, p=_pi[annotator_index, real_labels[i]])
        return real_labels, crowd_labels

    def fast_sample(self, p, _pi, I, num_annotators):
        #print("I:", I)
        #print("Num annotators:", num_annotators)
        J = len(p)
        K = _pi.shape[0]
        # Sample the real labels
        real_labels = np.random.choice(J, size=I, p=p)
        # Sample the annotators
        annotators = np.random.choice(K, size=(I, num_annotators))
        labels_and_annotators = annotators + real_labels[:, np.newaxis] * K
        labels_and_annotators = labels_and_annotators.flatten()
        unique_la, inverse_la, counts_la = np.unique(labels_and_annotators, return_inverse=True, return_counts=True)
        #print(inverse_la.shape)
        #print(inverse_la)
        crowd_labels = np.zeros((I * num_annotators, 3), dtype=np.int32)
        crowd_labels[:, 0] = np.arange(I * num_annotators) // num_annotators
        #crowd_labels.flatten()
        for i_la, label_and_annotator in enumerate(unique_la):
            real_label = label_and_annotator // K
            annotator_index = label_and_annotator % K
            #print("Real_label:", real_label)
            #print("Annotator:", annotator_index)
            emission_p = _pi[annotator_index, real_label]
            #print("i_la:", i_la)
            #print("counts:", counts_la[i_la])
            emitted_labels = np.random.choice(J, size=counts_la[i_la], p=emission_p)
            ca_indexes = np.equal(inverse_la, i_la)
            #print(ca_indexes.shape)
            #print(ca_indexes)
            crowd_labels[:, 1][ca_indexes] = annotator_index
            crowd_labels[:, 2][ca_indexes] = emitted_labels
        return real_labels, crowd_labels


    def majority_success_rate(self, p, _pi, I, K):
        real_labels, crowd_labels = self.fast_sample(p, _pi, I, K)
        # print(crowd_labels)
        c = majority_voting(crowd_labels, I, len(p))
        num_successes = np.sum(real_labels == c)
        return num_successes / I

    def majority_success_rates(self, p, _pi, I, annotators):
        success_p = np.zeros(len(annotators))
        for K in annotators:
            success_p[annotators.index(K)] = self.majority_success_rate(p, _pi, I, K)
        return success_p

    def DS_consensus_success_rate(self, p, _pi, I, K):
        real_labels, crowd_labels = self.fast_sample(p, _pi, I, K)
        # print(crowd_labels)
        prob_consensus, consensus = self.compute_consensus_with_fixed_parameters(p, _pi, crowd_labels, I)
        num_successes = np.sum(real_labels == consensus)
        return num_successes / I

    def DS_consensus_success_rates(self, p, _pi, I, annotators):
        success_p = np.zeros(len(annotators))
        for K in annotators:
            success_p[annotators.index(K)] = self.DS_consensus_success_rate(p, _pi, I, K)
        return success_p

    def compute_consensus_with_fixed_parameters(self, p, _pi, labels, I):
        self.I = I
        self.J = len(p)
        self.K = _pi.shape[0]
        n = self.compute_n(labels)
        logpi = np.log(_pi)
        prob_consensus = self.e_step(n, logpi, p)
        consensus = np.argmax(prob_consensus, axis=1)
        return prob_consensus, consensus

def prob_consensus(a, alpha=0.01, response=2):
    c = a.copy()
    # c = np.sort(a, axis=0)
    tasks, c[: ,0] = np.unique(c[: ,0], return_inverse=True)
    # print(c[:5,0])
    N = c.shape[0]
    # print("N=", N)
    n = len(tasks)
    # print("n=", n)
    labels, c[: ,response] = np.unique(c[: ,response], return_inverse=True)
    k = len(labels)
    responses = c[: ,response]
    r ,rc = np.unique(responses, return_counts=True)
    # print(r, rc)
    # print("k=", k, " Labels:", labels)
    # print(c[:,response])
    _pi = np.zeros(k)
    _pi[:] = rc[:]
    _pi /= np.sum(_pi)
    # print("pi:", _pi)
    # print("alpha:", alpha)
    q = np.zeros((n ,k))
    q += _pi[np.newaxis ,:]
    # print(q)
    # while ()


    old_q = np.zeros((n ,k))
    tol = 1e-6
    while not np.allclose(old_q ,q ,atol=tol):
        # print("pi:", _pi)
        # print("alpha:", alpha)

        P = np.ones((k ,k))
        P *= alpha
        np.fill_diagonal(P , 1 -alpha *( k -1))
        old_q[: ,:] = q
        q[: ,:] = _pi[np.newaxis ,:]
        for i_tr in range(N):
            q[c[i_tr, 0], :] *= P[c[i_tr, response]]
        sums = np.sum(q ,axis=1)
        q /= sums[:, np.newaxis]
        # print(q[:5,:]) B=0
        for i_tr in range(N):
            B += 1-q[c[i_tr , 0], c[i_tr, response]]
        alpha = B/(N*(k-1) )

        _pi = np.sum(q, axis=0)+1
        _pi /= np.sum(_pi)
    return q, _pi, alpha