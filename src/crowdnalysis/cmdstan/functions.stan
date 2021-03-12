
vector[] log_transpose(vector[] m) {
    int k = size(m);
    vector[k] log_m_t[k];
    for (_real in 1:k)
        for (_emitted in 1:k)
            log_m_t[_emitted,_real] = log(m[_real,_emitted]);
    return log_m_t;
}