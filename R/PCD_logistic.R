#logistic model
#use potential outcome to decide the true optimal treatment
library(simstudy)
library(data.table)
library(cmdstanr)
library(posterior)
library(slurmR)
library(dplyr)
library(ggplot2)
set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
univt_mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/logistic.stan");

#univt_mod <- cmdstan_model("./logistic.stan");
s_generate <- function(n_train = 500) {
  #---generate training data---#
  
  
  def <- defData(varname = "A", formula = "1;1", dist = "trtAssign")
  def <- defData(def, varname="cov_1", formula = 0, 
                 variance= 1, dist="normal") 
  def <- defData(def, varname = "y", dist = "binary",
                 formula = "0.35*cov_1 + A*(-0.4 + 0.2*cov_1)",
                 link = "logit")
  #---Generate data---#
  dd <- genData(n_train, def)
  return(dd)
}

univt_model <- function(generated_data, univt_mod){
  #set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")

  
  studydata <- list(
    N = nrow(generated_data), y=generated_data$y, 
    x=generated_data$cov_1,A=generated_data$A)
  
  fit_univt <- univt_mod$sample(
    data = studydata,
    refresh = 0,
    chains = 4L,
    parallel_chains = 4L,
    iter_warmup = 500,
    iter_sampling = 2500,
    show_messages = FALSE)
  
  diagnostics_df <- as_draws_df(fit_univt$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##
  # Get posterior draws of all parameters
  draws_dt <- data.table(as_draws_df(fit_univt$draws()))
  
  
  beta_trt <- draws_dt$beta_trt
  beta_inter_cov1 <- draws_dt$beta_inter

  
  res_univt <- data.table(beta_trt,beta_inter_cov1,
                    div,tree_hit)
  
  return(res_univt)
  
}
s_train <- function(n_train = 500, univt_mod=univt_mod){
  #---train the model---#
  generated_data <- s_generate(n_train = n_train)
  univt_results <- univt_model(generated_data,univt_mod)
  
  
  return(univt_results =univt_results)
}

opt_univt_trt <- function(cov, coefs,thresh = 0){
  ##---Given patients characteristics, derive tbi for each patients using single model for one outcome---##
  
  beta_trt <- coefs$beta_trt #main effect of trt
  #fixed effect in the interaction term
  beta_inter <- as.matrix(coefs[,c("beta_inter_cov1")])
  
  hte.distr <- apply(as.matrix(cov),1, function(x) beta_trt + beta_inter*x)
  prob.tbi.tmp <- apply(hte.distr,2, function(x) x < thresh)
  tbi <- apply(prob.tbi.tmp, 2, mean)
  opt_univt_trt  <- sapply(tbi, function(x) ifelse(x > 0.5, 1, 0)) #if Pr(tbi < 0) greater than 0.5, recommend CCP
  
  return(opt_univt_trt)
}

s_test_generate <- function(n_test = 2000) {
  #---generate test data and their potential outcomes---#
  
  def <- defData(varname="cov_1", formula = 0, 
                 variance= 1, dist="normal") 

  def <- defData(def, varname = "y_trt1", dist="binary",
                 formula = "0.35*cov_1 - 0.4*1 + 0.2*cov_1", link="logit")
 
  def <- defData(def, varname = "y_trt0", dist="binary",
                 formula = "0.35*cov_1",
                 link="logit")
  #the optimsl treatment decision based on potential outcomes
  def <- defData(def, varname = "optimal_trt",
                 formula = "1*(y_trt0 > y_trt1)",
                 dist="nonrandom")
  
  def <- defData(def, varname = "y_diff",
                     formula = "y_trt1 - y_trt0",
                     dist="nonrandom")
  #---Generate data---#
  dd <- genData(n_test, def)
  return(dd)
}

PCD <- function(train_coef,thresh=0,
                n_test = n_test
                ){
  ##---Proportion of correct decisions on test data---##
  test_data <- s_test_generate(n_test = n_test)
  ##Given patients characteristics, derive optimal treatment for each patients using the true value in data generating process
  cov <- test_data$cov_1
  
  trt_true <- test_data$optimal_trt
  
  trt_univt <- opt_univt_trt(cov=cov, coefs = train_coef,thresh = 0)
  
  #optimal decision based on logOR
  theta_diff <- -0.4 + 0.2%*%cov
  
  trt_true_or <- as.vector(ifelse(theta_diff >=0,0,1))
  
  test_data[ , `:=` (trt_univt= trt_univt,trt_true_or= trt_true_or)]
  
  PCD_by_y <- test_data[,
            .(pcd_by_ydiff = mean(optimal_trt==trt_univt),pcd_by_or = mean(trt_true_or == trt_univt)),
            keyby = .(y_diff)]
  
  
  return(PCD_by_y)
  
}

bayes_single_rep <- function(iter,
                              n_train = 500,
                             n_test=2000,univt_mod=univt_mod,thresh=0) {
  
  train_coef <- s_train(n_train = n_train,univt_mod=univt_mod)
  
  test_PCD <- PCD(train_coef=train_coef,thresh=0,
                  n_test = n_test) 
  
  div_univt <- mean(train_coef$div)
  return(data.table(iter,
                    test_PCD,
                    div_univt))
                    
}
# Run iterations on local laptop
#bayes_result <- rbindlist(lapply(1:2, function(x) bayes_single_rep(x,
#                                                         n_train = 500, n_test=2000,univt_mod=univt_mod)))


job <- Slurm_lapply(
  X = 1:100,
  FUN =bayes_single_rep,
  n_train = 500, n_test=2000,univt_mod=univt_mod,
  njobs = 60,
  mc.cores = 4L,
  job_name = "mvo_65",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "4:00:00", partition = "cpu_dev", `mem-per-cpu` = "8G"),
  export = c("s_generate", "univt_model","s_train","opt_univt_trt",
             "s_test_generate","PCD"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/logistic.rda")

####--plot---#####

# 

bayes_result <- res[res$div_univt <= 100, ]
dim(bayes_result)
#bayes_result$y_diff_abs <- 0:10
#PCD<- bayes_result[,c("iter","y_diff_abs","mvo_PCD","univt_PCD")]

PCD<- bayes_result[,c("iter","y_diff","pcd_by_ydiff","pcd_by_or")]

m_data <- reshape2::melt(PCD,id=c("iter","y_diff"))
m_data$y_diff <- as.factor(m_data$y_diff)
ggplot(m_data, aes(x=y_diff, y=value,fill= variable)) +
  geom_boxplot()+
  labs(title="PCD grouped by value of potential outcomes (n_train=500)",
       y = "PCD", x= "(difference in potential outcomes)")+
  labs(fill= "Criteria")
