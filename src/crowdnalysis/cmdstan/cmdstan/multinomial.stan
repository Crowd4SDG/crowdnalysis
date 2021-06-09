
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