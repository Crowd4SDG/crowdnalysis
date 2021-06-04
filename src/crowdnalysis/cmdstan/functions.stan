real square_distance(vector p, vector q) {
    return dot_product(p-q, p-q);
}

real kl(vector p, vector q) {
    return dot_product(p , (log(p) - log(q)));
}

real jsd(vector p, vector q) {
    return 0.5*kl(p,q)+0.5*kl(q,p);
}

vector[] log_transpose(vector[] m) {
    int k = size(m);
    int l = size(m[1]);
    vector[k] log_m_t[l];
    for (_real in 1:k)
        for (_emitted in 1:l)
            log_m_t[_emitted,_real] = log(m[_real,_emitted]);
    return log_m_t;
}

vector[] bounded_log_transpose(vector[] m) {
    int k = size(m);
    vector[k] log_m_t[k];
    for (_real in 1:k)
        for (_emitted in 1:k)
            log_m_t[_emitted,_real] = fmax(-100,log(m[_real,_emitted]));
    return log_m_t;
}

vector[] multinomial_log_p_t_C(vector tau, vector[] pi, int t, int[] t_A, int[] ann) {
    int k = size(tau);
    int l = size(pi[1]);
    //int t = size(log_emission_t);
    int a = size(t_A);

    print("l=",l);
    print("k=",k);
    // Make the log and transpose the emission matrix
    vector[k] log_emission_t[l];

    log_emission_t = log_transpose(pi);
    print("LogET", log_emission_t);

    vector[k] log_p_t_C[t];
    // Initialize with the prior

    log_p_t_C = rep_array(log(tau), t);

    // Update each task with the information contributed by its annotations

    for (_a in 1:a)
        log_p_t_C[t_A[_a]] += log_emission_t[ann[_a]];

    return log_p_t_C;
}


vector soften(vector v, real eps) {
    int size_v = size(v);
    vector[size_v] v_s = v;
    v_s += eps;
    v_s /= sum(v_s);
    return v_s;
}

vector[] bounded_multinomial_log_p_t_C(vector tau, vector[] pi, int t, int[] t_A, int[] ann) {
    int k = size(tau);
    int a = size(t_A);

    //vector[k] soft_tau = soften(tau, 0.01);
    //print("Soft tau:", soft_tau);
    //vector[k] soft_pi[k];
    //for (_k in 1:k) {
    //    soft_pi[_k] = soften(pi[_k], 0.1);
    //}
    //print("Soft pi:", soft_pi);

    // Make the log and transpose the emission matrix
    vector[k] log_emission_t[k];

    log_emission_t = log_transpose(pi);

    vector[k] log_p_t_C[t];
    // Initialize with the prior

    log_p_t_C = rep_array(log(tau), t);

    // Update each task with the information contributed by its annotations

    for (_a in 1:a)
        log_p_t_C[t_A[_a]] += log_emission_t[ann[_a]];

    return log_p_t_C;
}


vector[] ds_log_p_t_C(vector tau, vector[,] pi, int t, int[] t_A, int[] w_A, int[] ann) {
    int k = size(tau);
    int w = size(pi);
    int a = size(t_A);

    // Make the log and transpose the emission matrix
    vector[k] log_emission_t[w,k];
    for (_w in 1:w) {
        log_emission_t[_w] = log_transpose(pi[_w]);
    }

    vector[k] log_p_t_C[t];

    // Initialize with the prior

    log_p_t_C = rep_array(log(tau), t);

    // Update each task with the information contributed by its annotations

    for (_a in 1:a)
        log_p_t_C[t_A[_a]] += log_emission_t[w_A[_a]][ann[_a]];

    return log_p_t_C;
}

int [,] compute_movements(int k) {
    int dst[k,k-1];
    for (_k in 1:k) {
        for (_i in 1:k-1) {
            dst[_k][_i] = _i + (_i>=_k);
        }
    }
    // print(dst);
    return dst;
}

vector[] softmax_diag(vector[] eta, int[,] dst) {
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

//matrix log_transpose_m( matrix m) {
//   int k = size(m);
//    matrix log_m_t[k,k];
//    for (_real in 1:k)
//        for (_emitted in 1:k)
//            log_m_t[_emitted,_real] = log(m[_real,_emitted]);
//    return log_m_t;
//}