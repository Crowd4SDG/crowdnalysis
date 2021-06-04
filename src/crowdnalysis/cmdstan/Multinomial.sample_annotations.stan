data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> n_annotations_per_task; //number of annotations per task
  
  int<lower=2> k; //number of classes
  int<lower=2> l; //number of labels
  int <lower=1,upper=k> t_C[t];
}

transformed data {
  int a;
  a = t * n_annotations_per_task;
}

parameters {
  simplex[k] tau;
  simplex[l] pi[k];
}

generated quantities {


  // Generate annotations
  
  int<lower=1,upper=t> t_A[a]; // the item the n-th annotation belongs to
  int<lower=1,upper=w> w_A[a]; // the annotator which produced the n-th annotation
  int<lower=1,upper=l> ann[a]; // the annotation
  for (_t in 1:t) {
      for (_i in 1:n_annotations_per_task) {
        int _a;
        _a = (_t-1) * n_annotations_per_task + _i;
        t_A[_a] = _t;
        w_A[_a] = categorical_rng(rep_vector(1.0/w,w));
        ann[_a] = categorical_rng(pi[t_C[t_A[_a]]]);
      }
  }
}
