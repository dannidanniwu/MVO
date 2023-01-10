//compared to version_1
//sigma sd=3
//change the way reparameterization
data {
  int<lower=0> N;  //of observations
  int<lower=0> D; //  of binary outcome
  int<lower=2> L;
  int<lower=0> P;// of covariates
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=0,upper=L> y_ord[N];
  int<lower=0,upper=1> y_b[N,D];
  row_vector[P] x[N];       //covariates
}

parameters {
  real z_beta_0;
  matrix[P,(D+1)] z_beta_1;
  matrix[(P+1),(D+1)] z_beta_int;
  vector[(P+1)] z_beta_star;
  ordered[L-1] tau;
  vector<lower=0>[(P+1)] sigma_beta;
}

transformed parameters {
  vector[D] yhat_b[N];
  real yhat_ord[N];
  real beta_0;
  matrix[P,(D+1)] beta_1;
  matrix[(P+1),(D+1)] beta_int;
  vector[(P+1)] beta_star;
  
  //Non-Centered Parameterization
  beta_0 = 10*z_beta_0;
  beta_star = 10* z_beta_star;
  
  for (j in 1:P){
      beta_1[j] = 10*z_beta_1[j];
   }
   
  
  for (j in 1:(P+1))
     for (d in 1:(D+1)){
       beta_int[j,d] = beta_star[j] + sigma_beta[j]*z_beta_int[j,d];
  }
   
  for (i in 1:N){
         for (k in 1:D){
         yhat_b[i,k] =  beta_0 + x[i]*col(beta_1,k)  +  append_col(1, x[i]) * col(beta_int,k) * A[i];
    }
         yhat_ord[i] = x[i]*col(beta_1,D+1)  +  append_col(1, x[i]) * col(beta_int,D+1) * A[i];
  }
}

model{
  sigma_beta ~ exponential(10);
  
  for (l in 1: (L-1)){
    tau[l] ~ student_t(3,0,8);
  }
  
  to_vector(z_beta_int) ~ std_normal();
  
  for (i in 1:N){
       y_ord[i] ~ ordered_logistic(yhat_ord[i],tau);
    for (k in 1:D){
       y_b[i,k] ~ bernoulli_logit(yhat_b[i,k]);
    }
  }
}

