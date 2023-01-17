data {
  int<lower=0> N;  //of observations
  int<lower=0> D; //  of binary outcome
  int<lower=2> L;
  int<lower=0> P;// of covariates
  int<lower= 0,upper= 1> A[N];  //treatment assignment
  int<lower=1,upper=L> y_ord[N];
  int<lower=0,upper=1> y_b[N,D];
  row_vector[P] x[N];       //covariates
}

parameters {
  vector[D] beta_0;
  matrix[P,(D+1)] z_beta_1;
  matrix[(P+1),(D+1)] z_beta_ind[N];
  matrix[(P+1),(D+1)] z_beta_int;
  vector[(P+1)] z_beta_star;
  ordered[L-1] tau;
  vector<lower=0>[(P+1)] sigma_beta;
  vector<lower=0>[(P+1)] sigma;
  cholesky_factor_corr[(D+1)] L_Phi[(P+1)];
}

transformed parameters {
  vector[D] yhat_b[N];
  real yhat_ord[N];
  matrix[P,(D+1)] beta_1;
  matrix[(P+1),(D+1)] beta_ind[N];
  matrix[(P+1),(D+1)] beta_int;
  vector[(P+1)] beta_star;
  
  //Non-Centered Parameterization
  beta_star = 10* z_beta_star;
  
  for (j in 1:P){
      beta_1[j] = 10*z_beta_1[j];
   }
   
  for (j in 1:(P+1))
     for (d in 1:(D+1)){
       beta_int[j,d] = beta_star[j] + sigma_beta[j]*z_beta_int[j,d];
  }
   
  for (i in 1:N){
      for(j in 1:(P+1)){
        beta_ind[i,j] = beta_int[j] + z_beta_ind[i,j]*diag_pre_multiply(rep_vector(sigma[j],D+1),L_Phi[j]);
       }
         for (k in 1:D){
         yhat_b[i,k] =  beta_0[k] + x[i]*col(beta_1,k)  +  append_col(1, x[i]) * col(beta_ind[i],k) * A[i];
    }
         yhat_ord[i] = x[i]*col(beta_1,D+1)  +  append_col(1, x[i]) * col(beta_ind[i],D+1) * A[i];
  }
}

model {
  
 
  sigma ~ exponential(100);
  sigma_beta ~ exponential(10);
  for ( j in 1:(P+1)){
    L_Phi[j] ~ lkj_corr_cholesky(1);
  }
  for (l in 1: (L-1)){
    tau[l] ~ student_t(3,0,8);
  }
  
  to_vector(z_beta_int) ~ std_normal();
  
  for (i in 1:N){
    to_vector(z_beta_ind[i]) ~ std_normal();
    y_ord[i] ~ ordered_logistic(yhat_ord[i],tau);
    for (k in 1:D){
       beta_0[k] ~ student_t(3,0,8);
       y_b[i,k] ~ bernoulli_logit(yhat_b[i,k]);
    }
  }
}

generated quantities {
  corr_matrix[(D+1)] Phi_c[(P+1)];
  for (j in 1:(P+1)){
    Phi_c[j]= L_Phi[j] * L_Phi[j]';
  }
}
