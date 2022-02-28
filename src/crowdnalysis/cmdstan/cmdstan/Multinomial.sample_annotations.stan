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
  for (t_ in 1:t) {
      for (i_ in 1:n_annotations_per_task) {
        int a_;
        a_ = (t_-1) * n_annotations_per_task + i_;
        t_A[a_] = t_;
        w_A[a_] = categorical_rng(rep_vector(1.0/w,w));
        ann[a_] = categorical_rng(pi[t_C[t_A[a_]]]);
      }
  }
}
