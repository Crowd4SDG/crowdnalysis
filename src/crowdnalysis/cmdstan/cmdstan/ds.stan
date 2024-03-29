vector[,] ds_log_transpose(vector[,] pi) {
    int w = size(pi);
    int l = size(pi[1,1]);
    int k = size(pi[1]) / l;

    vector[k] log_pi_t[w,l];
    for (w_ in 1:w) {
        log_pi_t[w_] = log_transpose(pi[w_]);
    }
    return log_pi_t;
}


vector[] ds_log_p_t_C(vector tau, vector[,] pi, int t, int[] t_A, int[] w_A, int[] ann) {
    int k = size(tau);
    int w = size(pi);
    int l = size(pi[1,1]);
    int a = size(t_A);

    // Make the log and transpose the emission matrix
    vector[k] log_emission_t[w,l];
    log_emission_t = ds_log_transpose(pi);

    vector[k] log_p_t_C[t];

    // Initialize with the prior

    log_p_t_C = rep_array(log(tau), t);

    // Update each task with the information contributed by its annotations

    for (a_ in 1:a)
        log_p_t_C[t_A[a_]] += log_emission_t[w_A[a_]][ann[a_]];

    return log_p_t_C;
}

