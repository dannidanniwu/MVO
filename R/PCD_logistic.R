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
  set_cmdstan_path(path = "/gpfs/share/apps/cmdstan/2.25.0")

  
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

  #define canonical parameter for 4 outcomes: 
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
# bayes_result <- rbindlist(lapply(1:2, function(x) bayes_single_rep(x,
#                                                         n_train = 500, n_test=2000,univt_mod=univt_mod)))

# save(bayes_result,file="./PCD_compareITR_y_OR.rda")
# 
# apply(bayes_result,2,function(x) round(x,digits = 1))

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
#check proportion of levels of outcomes
# summary(res[, 97:113])
# bayes_result <- res[res$div_mvo <= 100, ]
# dim(bayes_result)#0.01:71/0.1:41/1:79
# #the ordinal outcome
# 
# 
# gener_y4 <- data.frame(iter=1,
# 
#                        beta_cov1_mvo= 0.35,
#                        beta_cov2_mvo = -0.4,
#                        beta_cov3_mvo = 0.15,
#                        beta_cov4_mvo = 0.2,
#                        beta_cov5_mvo = -0.21,
#                        beta_trt_mvo = 0.4,
#                        beta_inter_cov1_mvo=0.2,
#                        beta_inter_cov2_mvo=-0.1,
#                        beta_inter_cov3_mvo=0.1,
#                        beta_inter_cov4_mvo=0.05,
#                        beta_inter_cov5_mvo=-0.06,
#                        beta_star_trt_mvo=0.395,
#                        beta_star_cov1_mvo=0.195,
#                        beta_star_cov2_mvo=-0.105,
#                        beta_star_cov3_mvo=0.105,
#                        beta_star_cov4_mvo= 0.045,
#                        beta_star_cov5_mvo= -0.055,
#                        tau_1_mvo= -0.8,
#                        tau_2_mvo= 0.41,
#                        tau_3_mvo= 1.39,
#                        sigma_beta_cov1_mvo= NA,
#                        sigma_beta_cov2_mvo = NA,
#                        sigma_beta_cov3_mvo = NA,
#                        sigma_beta_cov4_mvo = NA,
#                        sigma_beta_cov5_mvo = NA,
#                        sigma_beta_trt_mvo = NA,
#                        Md="True value")
# M_MVO <- subset(bayes_result,select=c("iter","beta_cov1_mvo","beta_cov2_mvo","beta_cov3_mvo",
#                                       "beta_cov4_mvo","beta_cov5_mvo",
#                                       "beta_trt_mvo","beta_inter_cov1_mvo",
#                                       "beta_inter_cov2_mvo","beta_inter_cov3_mvo","beta_inter_cov4_mvo","beta_inter_cov5_mvo",
#                                       "tau_1_mvo","tau_2_mvo","tau_3_mvo","beta_star_trt_mvo",
#                                       "beta_star_cov1_mvo",
#                                       "beta_star_cov2_mvo","beta_star_cov3_mvo","beta_star_cov4_mvo","beta_star_cov5_mvo",
#                                       "sigma_beta_cov1_mvo","sigma_beta_cov2_mvo","sigma_beta_cov3_mvo",
#                                       "sigma_beta_cov4_mvo","sigma_beta_cov5_mvo","sigma_beta_trt_mvo" ))
# M_MVO$Md <- "MVO"
# 
# M_UNI <- subset(bayes_result,select=c("iter","beta_cov1_univt","beta_cov2_univt","beta_cov3_univt",
#                                       "beta_cov4_univt","beta_cov5_univt",
#                                       "beta_trt_univt","beta_inter_cov1_univt",
#                                       "beta_inter_cov2_univt","beta_inter_cov3_univt","beta_inter_cov4_univt","beta_inter_cov5_univt",
#                                       "tau_1_univt","tau_2_univt","tau_3_univt"))%>%mutate("beta_star_trt_univt"=NA,
#                                       "beta_star_cov1_univt"=NA,
#                                      "beta_star_cov2_univt"=NA,
#                                      "beta_star_cov3_univt"=NA,"beta_star_cov4_univt"=NA,"beta_star_cov5_univt"=NA,
# 
#                                       "sigma_beta_cov1_univt"=NA, "sigma_beta_cov2_univt"=NA,
#                                       "sigma_beta_cov3_univt"=NA, "sigma_beta_cov4_univt"=NA,
#                                       "sigma_beta_cov5_univt"=NA, "sigma_beta_trt_univt"=NA,"Md"="UNIO")
# colnames(M_UNI) <-c("iter","beta_cov1_mvo","beta_cov2_mvo","beta_cov3_mvo",
#                      "beta_cov4_mvo","beta_cov5_mvo",
#                      "beta_trt_mvo","beta_inter_cov1_mvo",
#                      "beta_inter_cov2_mvo","beta_inter_cov3_mvo","beta_inter_cov4_mvo","beta_inter_cov5_mvo",
#                      "tau_1_mvo","tau_2_mvo","tau_3_mvo","beta_star_trt_mvo",
#                      "beta_star_cov1_mvo",
#                      "beta_star_cov2_mvo","beta_star_cov3_mvo","beta_star_cov4_mvo","beta_star_cov5_mvo",
#                      "sigma_beta_cov1_mvo","sigma_beta_cov2_mvo","sigma_beta_cov3_mvo",
#                      "sigma_beta_cov4_mvo","sigma_beta_cov5_mvo", "sigma_beta_trt_mvo",
#                     "Md")
# 
# D_all <-rbind(M_UNI,M_MVO,gener_y4)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="Posterior mean of paramters (no individual beta's layer, outcome specific tau)",
#        y = "Posterior mean")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
# 
# 
# 
# # ##########sd#############
# SD_MVO <- subset(bayes_result,select=c("iter","beta_cov1_sd_mvo","beta_cov2_sd_mvo","beta_cov3_sd_mvo",
#                                       "beta_cov4_sd_mvo","beta_cov5_sd_mvo",
#                                       "beta_trt_sd_mvo","beta_inter_cov1_sd_mvo",
#                                       "beta_inter_cov2_sd_mvo","beta_inter_cov3_sd_mvo","beta_inter_cov4_sd_mvo","beta_inter_cov5_sd_mvo",
#                                       "tau_1_sd_mvo","tau_2_sd_mvo","tau_3_sd_mvo","beta_star_trt_sd_mvo",
#                                       "beta_star_cov1_sd_mvo",
#                                       "beta_star_cov2_sd_mvo","beta_star_cov3_sd_mvo","beta_star_cov4_sd_mvo","beta_star_cov5_sd_mvo"))
# SD_MVO$Md <- "MVO"
# 
# SD_UNI <- subset(bayes_result,select=c("iter","beta_cov1_sd_univt","beta_cov2_sd_univt","beta_cov3_sd_univt",
#                                       "beta_cov4_sd_univt","beta_cov5_sd_univt",
#                                       "beta_trt_sd_univt","beta_inter_cov1_sd_univt",
#                                       "beta_inter_cov2_sd_univt","beta_inter_cov3_sd_univt","beta_inter_cov4_sd_univt","beta_inter_cov5_sd_univt",
#                                       "tau_1_sd_univt","tau_2_sd_univt","tau_3_sd_univt"))%>%mutate("beta_star_trt_sd_univt"=NA,
#                                                                                            "beta_star_cov1_sd_univt"=NA,
#                                                                                            "beta_star_cov2_sd_univt"=NA,
#                                                                                            "beta_star_cov3_sd_univt"=NA,"beta_star_cov4_sd_univt"=NA,"beta_star_cov5_sd_univt"=NA,"Md"="UNIO")
# colnames(SD_UNI) <-c("iter","beta_cov1_sd_mvo","beta_cov2_sd_mvo","beta_cov3_sd_mvo",
#                      "beta_cov4_sd_mvo","beta_cov5_sd_mvo",
#                      "beta_trt_sd_mvo","beta_inter_cov1_sd_mvo",
#                      "beta_inter_cov2_sd_mvo","beta_inter_cov3_sd_mvo","beta_inter_cov4_sd_mvo","beta_inter_cov5_sd_mvo",
#                      "tau_1_sd_mvo","tau_2_sd_mvo","tau_3_sd_mvo","beta_star_trt_sd_mvo",
#                      "beta_star_cov1_sd_mvo",
#                      "beta_star_cov2_sd_mvo","beta_star_cov3_sd_mvo","beta_star_cov4_sd_mvo","beta_star_cov5_sd_mvo",
#                     "Md")
# 
# D_all <-rbind(SD_UNI,SD_MVO)
# 
# M_data <- reshape2::melt(D_all,id=c("iter","Md"))
# 
# ggplot(M_data, aes(x=variable, y=value,fill=Md)) +
#   geom_boxplot(width=0.25,position = position_dodge(width = 0.5))+ theme_minimal()+
#   labs(title="Posterior SD of paramters (no individual beta's layer, outcome specific tau)",
#        y = "Posterior SD")+facet_wrap(~variable,scale=c("free"), labeller = label_parsed)+
#   theme(strip.text.x = element_blank())+labs(fill="CLASS")
# 
# 
# #####PCD#########
# PCD<- bayes_result[,c("iter","mvo_PCD","univt_PCD")]
# 
# m_data <- reshape2::melt(PCD,id=c("iter"))
# 
# ggplot(m_data, aes(x=variable, y=value)) +
#   geom_boxplot()+
#   labs(title="PCD based on simualtions with low div (no individual beta's layer, outcome specific tau)",
#        y = "PCD", x= "model")+
#   labs(fill= "Model")
# 
# 
# #####PCD#########
# PCD$diff <- PCD$mvo_PCD -PCD$univt_PCD
# round(summary(PCD$diff),3)
# ggplot(PCD, aes(y=diff)) +
#   geom_boxplot()+
#   labs(title="PCD difference based on simualtions with low div (no individual beta's layer, outcome specific tau)",
#        y = "PCD difference")
# # 
