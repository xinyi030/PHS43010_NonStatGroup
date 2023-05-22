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
mu <- c(1, 1)
Sigma <- rbind(c(0.84^2, 0.134), c(0.134, 0.80^2))
bcrm.sim <- bcrm(stop=list(nmax=36), dose=dose, ff="logit2", p.tox0=truep1
, prior.alpha=list(4, mu, Sigma), target.tox=target.tox, constrain=TRUE, pointest="mean", 
start=1, simulate=TRUE, nsims=nsims, truep=truep1, method="rjags", sdose.calculate = "median")

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


    alpha.median <- median(posterior.samples[,1])
    beta.median <- median(posterior.samples[,2])
    post.tox <- logit2(bcrm.sim[[i]]$sdose, alpha.median, beta.median)
    mtd <- dose[which.min(abs(post.tox - 1/3))]
    mtds[i] <- mtd
}

table(mtds)/nsims
print(bcrm.sim)
plot(bcrm.sim, trajectories=FALSE)
plot(bcrm.sim, trajectories=TRUE)

# Check the prior predictive distribution
# define the dose label function
# dose.label <- function(p, alpha, beta){
#   return (log(p/(1-p))-log(alpha))/beta
# }

# post.dose.label <- dose.label(p.tox0, exp(3), 1)

# tests <- c(seq(from=post.dose.label[1], to=post.dose.label[2], length.out=1000),
# seq(from=post.dose.label[2], to=post.dose.label[3], length.out=1000),
# seq(from=post.dose.label[3], to=post.dose.label[4], length.out=1000),
# seq(from=post.dose.label[4], to=post.dose.label[5], length.out=1000))

# pred.tests <- numeric(4000)

# for(i in 1:4000) {
#     pred.tests[i] <- logit2(tests[i], exp(3), 1)
# }

# dose.tests <- c(seq(from=0.5, to=1, length.out=1000),
# seq(from=1, to=3, length.out=1000),
# seq(from=3, to=5, length.out=1000),
# seq(from=5, to=6, length.out=1000))

# res <- cbind(dose.tests, pred.tests)

# # plot(res, type="l", xlab="Dose", ylab="Probability of toxicity")

# points(dose, logit2(post.dose.label, 1, 1), col="red", pch=19, cex=2)
# points(dose, logit2(post.dose.label, 1, 1), type="l")