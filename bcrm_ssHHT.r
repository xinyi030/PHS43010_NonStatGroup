# install.packages("bcrm")
# install.packages("rjags")
# install.packages("R2WinBUGS")

library(bcrm)
library(rjags)
library(R2WinBUGS)

dose <- c(0.5, 1, 3, 5, 6)
truep1 <- c(0.25, 0.3, 0.5, 0.6, 0.7)
truep2 <- c(0.01, 0.05, 0.2, 0.3, 0.5)
p.tox0 <- c(0.05, 0.10, 0.15, 0.33,0.50) 
nsims <- 1000

# 1. Initialize the parameters of bcrm.
target.tox <- 1/3
mu <- c(2.15, 0.52)
Sigma <- rbind(c(0.84^2, 0.134), c(0.134, 0.80^2))
bcrm.sim <- bcrm(stop=list(nmax=36), dose=dose, ff="logit2", p.tox0 = p.tox0
, prior.alpha=list(4, mu, Sigma), target.tox=target.tox, constrain=TRUE, pointest="mean", 
start=1, simulate=TRUE, nsims=nsims, truep=truep1, method="rjags")

# define a two-parameter logistic function
logit2 <- function(x, alpha, beta) {
  return(exp(log(alpha)+beta*x)/(1+exp(log(alpha)+beta*x)))
}

# 3. Find the MTD of each simulation.
# a loop from 1 to 1000.
mtds <- numeric(nsims)
for (i in 1:nsims) {
   posterior.samples <- Posterior.rjags(bcrm.sim[[i]]$tox, bcrm.sim[[i]]$notox, bcrm.sim[[i]]$sdose,
    ff = "logit2", prior.alpha = list(4, mu, Sigma), burnin.itr = 2000, production.itr = 2000)


    alpha.mean <- mean(posterior.samples[,1])
    beta.mean <- mean(posterior.samples[,2])
    post.tox <- logit2(bcrm.sim[[i]]$sdose, alpha.mean, beta.mean)
    mtd <- dose[which.min(abs(post.tox - target.tox))]
    mtds[i] <- mtd
}

table(mtds)/1000
print(bcrm.sim)
plot(bcrm.sim, trajectories=FALSE,  file="bcrm_ssHHT_scenario1_summary.pdf")
plot(bcrm.sim, trajectories=TRUE,  file="bcrm_ssHHT_scenario1_trajectories.pdf")
