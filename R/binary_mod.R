library(simstudy)
library(data.table)
library(cmdstanr)
library(posterior)
library(slurmR)
library(dplyr)
library(ggplot2)
set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
mod <- cmdstan_model("/gpfs/data/troxellab/danniw/r/binary_mod_stan.stan");
s_generate <- function(beta_0=-0.7, beta_1=0.5,beta_2=0.4, beta_3=0.2,#controls the contribution from the two sources of variation. 
                       sigma_beta_0=0.2, sigma_beta_1=0.1,
                       n_train = 400) {
  
  def <- defData(varname = "A", formula = "1;1", dist = "trtAssign")
  def <- defData(def, varname="cov_1", formula = "0", 
                 variance= 1, dist="normal")  
  def <- defData(def, varname="beta_0_ind",formula = "0", 
                 variance= "..sigma_beta_0", dist="normal") 
  def <- defData(def, varname="beta_1_ind",formula = "0", 
                 variance= "..sigma_beta_1", dist="normal")  
  
  #define canonical parameter for 3 outcomes: 
  def <- defData(def, varname = "y_1",dist="binary",  
                 formula = "..beta_0 + ..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind + ..beta_3*cov_1 + beta_1_ind*cov_1) ",
                  link="logit")
  def <- defData(def,varname = "y_2",dist="binary",   
                 formula = "..beta_0 + ..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind  + ..beta_3*cov_1 + beta_1_ind*cov_1)",
                  link="logit")
  def <- defData(def,varname = "y_3", dist="binary",  
                 formula = "..beta_0 + ..beta_1*cov_1 +
                      A*(..beta_2 + beta_0_ind + ..beta_3*cov_1 + beta_1_ind*cov_1)",
                 link="logit")
  #---Generate data---#
  ds <- genData(n_train, def)
}

mvo_model <- function(iter, generated_data,mod){
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")
  x <- model.matrix( ~ cov_1, data = generated_data)[,-1]
  y <- generated_data[,c("y_1","y_2","y_3")]
  
  D <- 3
  studydata <- list(
    N = nrow(generated_data), y=y,
    x=x,A=generated_data$A, D=D)
  #J: #outcome
  #D: # of cov
  
  fit <- mod$sample(
    data = studydata,
    refresh = 0,
    chains = 4L,
    parallel_chains = 4L,
    iter_warmup = 500,
    iter_sampling = 2500,
    show_messages = FALSE,
    adapt_delta = 0.9)
  
  diagnostics_df <- as_draws_df(fit$sampler_diagnostics())
  div <- sum(diagnostics_df[, 'divergent__'])
  tree_hit <- sum(diagnostics_df$treedepth__ == 10)
  ##Get posterior draws of all parameters
  draws_dt <- data.frame(as_draws_df(fit$draws()))
  
  beta_0 <- draws_dt$beta_0#main effect of treatment
  #fixed effect in the interaction term for the first outcome
  beta_1 <- draws_dt$beta_1 
  beta_2 <- draws_dt$beta_2
  beta_3 <- draws_dt$beta_3
  sigma_beta_0 <- draws_dt$sigma_beta_0
  sigma_beta_1 <- draws_dt$sigma_beta_1

  
  res <- data.table(beta_0,beta_1,beta_2,beta_3,
                    sigma_beta_0,sigma_beta_1,
                    div,tree_hit)
  res_m <- round(apply(res,2,mean),3)
  
  y_eq1_p <- apply(y,2,function(x) sum(x)/length(x))
  return(data.table(iter=iter,beta_0=res_m[1],beta_1=res_m[2],
                    beta_2=res_m[3],beta_3=res_m[4],
                    sigma_beta_0=res_m[5],sigma_beta_1=res_m[6],
                    div=res_m[7],tree_hit=res_m[8],
                    y_1_p=y_eq1_p[1],
                    y_2_p=y_eq1_p[2],
                    y_3_p=y_eq1_p[3]
                    ))
}
s_train <- function(iter,beta_0=-0.7, beta_1=0.5,beta_2=0.4, beta_3=0.2,#controls the contribution from the two sources of variation. 
                    sigma_beta_0=0.2, sigma_beta_1=0.1,
                    n_train = 400,mod){
  generated_data <- s_generate(beta_0=beta_0, beta_1=beta_1,beta_2=beta_2,beta_3=beta_3,#controls the contribution from the two sources of variation. 
                               sigma_beta_0=sigma_beta_0, sigma_beta_1=sigma_beta_1,
                               n_train = n_train)
  
  #mvo model
  mvo_results <- mvo_model(iter=1,generated_data,mod)
  return(mvo_results)
}
# bayes_result <- rbindlist(lapply(1:20, function(x) s_train(x,beta_0=1, beta_1=2,beta_2=3, beta_3=2.5,#controls the contribution from the two sources of variation. 
#                                                            sigma_beta_0=0.5, sigma_beta_1=0.7,
#                                                            n_train = 400,mod=mod)))
# 
# save(bayes_result,file="./bayes_result.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

job <- Slurm_lapply(
  X = 1:100,
  FUN = s_train,
  beta_0= -0.7, 
  beta_1=0.5,
  beta_2=0.4, 
  beta_3=0.2,#controls the contribution from the two sources of variation. 
  sigma_beta_0=0.2, 
  sigma_beta_1=0.1,
  n_train = 400,mod=mod,
  njobs = 25,
  mc.cores = 4L,
  job_name = "mvo_5",
  tmp_path = "/gpfs/data/troxellab/danniw/scratch",
  plan = "wait",
  sbatch_opt = list(time = "4:00:00", partition = "cpu_dev", `mem-per-cpu` = "5G"),
  export = c("s_generate", "mvo_model"),
  overwrite = TRUE
)


res <- Slurm_collect(job)
res <- rbindlist(res)
save(res, file = "/gpfs/data/troxellab/danniw/data/binary_model_res_v2.rda")




####--plot---#####
# bayes_result <- res
# gener_y1 <- data.frame(iter=1,
#                        beta_0=1, beta_1=2,beta_2=3, beta_3=2.5,#controls the contribution from the two sources of variation.
#                        sigma_beta_0=0.5, sigma_beta_1=0.7,
#                        Md="True value")
# 
# M_MVO <- subset(bayes_result,select=-c(y_1_p, y_2_p,  y_3_p,div, tree_hit))
# M_MVO$Md <- "Estimated"
# D_all <-rbind(M_MVO,gener_y1)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="MVO: Posterior mean of paramters",
#        y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
