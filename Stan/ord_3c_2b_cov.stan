//compared to version_1
//sigma sd=3
//change the way reparameterization
data {
  int<lower=0> N;  //of observations
  int<lower=2> L;
  int<lower=0> P;// of covariates
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=0,upper=L> y_ord[N];
  row_vector[P] x[N];       //covariates
}

parameters {
  vector[P] z_beta_1;
  vector[(P+1)] z_beta_inter;
  ordered[L-1] tau;
}

transformed parameters {
  real yhat_ord[N];
  vector[P] beta_1;
  vector[(P+1)] beta_inter;
  
  //Non-Centered Parameterization
  beta_1 = 10*z_beta_1;
  beta_inter = 10*z_beta_inter;
   
  for (i in 1:N){
        yhat_ord[i] = x[i]*beta_1  +  append_col(1, x[i]) * beta_inter * A[i];
    }
}

model {

  for (l in 1: (L-1)){
    tau[l] ~ student_t(3,0,8);
  }
  
  z_beta_inter ~ std_normal(); 
  z_beta_1 ~ std_normal(); 
  
  for (i in 1:N){
    y_ord[i] ~ ordered_logistic(yhat_ord[i],tau);
  }
}


