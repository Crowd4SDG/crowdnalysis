data {
  int<lower=1> J; //number of annotators
  int<lower=2> K; //number of classes
  int<lower=1> N; //number of annotations
  int<lower=1> I; //number of items
  int<lower=1,upper=I> ii[N]; //the item the n-th annotation belongs to
  int<lower=1,upper=J> jj[N]; //the annotator which produced the n-th annotation
  int y[N]; //the class of the n-th annotation
}

transformed data
{
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
  vector[K] zeros = rep_vector(0,K);
}

parameters {
  simplex[K] pi;
  matrix[K,K-1] beta_raw[J];
  matrix[K,K-1] zeta;
  matrix<lower=0>[K,K-1] Omega;
}

transformed parameters 
{
  matrix[K,K-1] beta[J];
  matrix[K,K] beta_norm[J];
  vector[K] log_q_c[I];
  vector[K] log_pi;
  
  for(j in 1:J)
  {
    beta[j] = zeta + Omega .* beta_raw[j]; //non centered parameterization
    beta_norm[j] = append_col(beta[j], zeros); //fix last category to 0 (softmax non-identifiability)
    for(h in 1:K)
      beta_norm[j,h] = beta_norm[j,h] - log_sum_exp(beta_norm[j,h]); //log_softmax
  }
  
  log_pi = log(pi);
  for (i in 1:I)
    log_q_c[i] = log_pi;
     
  for (n in 1:N)
   for (h in 1:K)
     log_q_c[ii[n],h] = log_q_c[ii[n],h] + beta_norm[jj[n],h,y[n]];
}

model {
  
  pi ~ dirichlet(alpha);
  to_vector(zeta) ~ normal(0,1);
  to_vector(Omega) ~ normal(0,1);
    
  for(j in 1:J)
    to_vector(beta_raw[j]) ~ normal(0, 1); //part of the non centered parameterization
      
  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  vector[K] q_z[I]; //the true class distribution of each item
  
  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
