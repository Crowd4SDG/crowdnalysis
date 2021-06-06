
int [,] compute_movements(int l, int[] classes) {
    int k=size(classes);
    int dst[k,l-1];
    for (_k in 1:k) {
        for (_l in 1:l-1) {
            dst[_k][_l] = _l + (_l>=classes[_k]);
        }
    }
    print(dst);
    return dst;
}

int [,] compute_movements_old(int k) {
    int dst[k,k-1];
    for (_k in 1:k) {
        for (_i in 1:k-1) {
            dst[_k][_i] = _i + (_i>=_k);
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
    for (_k in 1:k) {
        pi[_k][dst[_k]] = -eta[_k];
        pi[_k][classes[_k]] = 0.;
        pi[_k] = softmax(to_vector(pi[_k]));
    }
    return pi;
}

vector[] softmax_diag_old(vector[] eta, int[,] dst) {
    int k = size(eta);
    vector[k] pi[k];
    // print("eta",eta);
    for (_k in 1:k) {
        pi[_k][dst[_k]] = -eta[_k];
        pi[_k][_k] = 0.;
        pi[_k] = softmax(to_vector(pi[_k]));
    }
    return pi;
}