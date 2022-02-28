
int [,] compute_movements(int l, int[] classes) {
    int k=size(classes);
    int dst[k,l-1];
    for (k_ in 1:k) {
        for (l_ in 1:l-1) {
            dst[k_][l_] = l_ + (l_>=classes[k_]);
        }
    }
    print(dst);
    return dst;
}

int [,] compute_movements_old(int k) {
    int dst[k,k-1];
    for (k_ in 1:k) {
        for (i_ in 1:k-1) {
            dst[k_][i_] = i_ + (i_>=k_);
        }
    }
    // print(dst);
    return dst;
}

vector[] softmax_diag(vector[] eta, int[] classes, int[,] dst) {
    int k = size(eta);
    int l = size(eta[1])+1;
    vector[l] pi[k];
    // print("eta",eta);
    for (k_ in 1:k) {
        pi[k_][dst[k_]] = -eta[k_];
        pi[k_][classes[k_]] = 0.;
        pi[k_] = softmax(to_vector(pi[k_]));
    }
    return pi;
}

vector[] softmax_diag_old(vector[] eta, int[,] dst) {
    int k = size(eta);
    vector[k] pi[k];
    // print("eta",eta);
    for (k_ in 1:k) {
        pi[k_][dst[k_]] = -eta[k_];
        pi[k_][k_] = 0.;
        pi[k_] = softmax(to_vector(pi[k_]));
    }
    return pi;
}