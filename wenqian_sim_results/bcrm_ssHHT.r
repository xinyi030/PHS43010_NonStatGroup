# install.packages("bcrm")
# install.packages("rjags")
# install.packages("R2WinBUGS")

library(bcrm)
library(rjags)
library(R2WinBUGS)
library(ggplot2)

dose <- c(0.5, 1, 3, 5, 6)
truep1 <- c(0.25, 0.3, 0.5, 0.6, 0.7)
truep2 <- c(0.01, 0.05, 0.2, 0.3, 0.5)
p.tox0 <- c(0.05, 0.10, 0.15, 0.33,0.50) 
nsims <- 1000

# 1. Initialize the parameters of bcrm.
target.tox <- 1/3
mu <- c(1, 1)
Sigma <- rbind(c(0.84^2, 0.134), c(0.134, 0.80^2))
bcrm.sim <- bcrm(stop=list(nmax=36), dose=dose, ff="logit2", p.tox0=p.tox0
, prior.alpha=list(4, mu, Sigma), target.tox=target.tox, constrain=TRUE, pointest=0.5, 
start=1, simulate=TRUE, nsims=nsims, truep=truep1, method="rjags", sdose.calculate = "median")

# define a two-parameter logistic function
logit2 <- function(x, alpha, beta) {
  return(exp(log(alpha)+beta*x)/(1+exp(log(alpha)+beta*x)))
}
yeah
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

# checking the variance of the posterior distribution
var(posterior.samples[,1])

# Check the prior predictive distribution
# define the dose label function
dose.label <- function(p, alpha, beta){
  return (log(p/(1-p))-log(alpha))/beta
}

post.dose.label <- dose.label(p.tox0, 1, 1)

tests <- seq(from=min(post.dose.label), to=max(post.dose.label), length.out=4000)

pred.tests <- numeric(4000)

for(i in 1:4000) {
    pred.tests[i] <- logit2(tests[i], 1, 1)
}

# dose.tests <- c(seq(from=0.5, to=1, length.out=1000),
# seq(from=1, to=3, length.out=1000),
# seq(from=3, to=5, length.out=1000),
# seq(from=5, to=6, length.out=1000))

res <- data.frame(tests, pred.tests)

prior_plot <- ggplot(data = res, aes(x = tests, y = pred.tests)) +
  geom_line() +
  labs(x = "Dose levels", y = "Probability of toxicity", title = "Prior predictive distribution") +
  scale_x_continuous(breaks = post.dose.label, labels = dose) +
  theme(text = element_text(size = 20))
prior_plot

ggsave("prior.png", prior_plot, width = 5, height = 6)

# points(dose, logit2(post.dose.label, 1, 1), col="red", pch=19, cex=2)
# points(dose, logit2(post.dose.label, 1, 1), type="l")

# Plot the two toxicity bars with ggplot and large fonts
# Make plot size as (12, 8)
tox_bar <- c(truep1, truep2)
dose_bar <- c(1:5, 1:5)
scenario_bar <- c(rep("Scenario 1", 5), rep("Scenario 2", 5))
plot_data <- data.frame(tox_bar, dose_bar, scenario_bar)
plot <- ggplot(data = plot_data, aes(x = dose_bar, y = tox_bar, fill = scenario_bar)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_abline(intercept = 1/3, slope = 0, linewidth = 2) +
  scale_x_continuous(breaks = seq(1, 5, 1), labels = dose) +
  labs(x = "Dose (mg/m^2/day)", y = "True Probability of toxicity", title = "Toxicity probability of two scenarios") +
  theme(text = element_text(size = 20))
# Adjust the width and height
width <- 12  # Width in inches
height <- 6  # Height in inches

# Save the plot with adjusted dimensions
ggsave("true_tox.png", plot, width = width, height = height)

patient_proportion1 <- c(0.301, 0.378, 0.276, 0.0427, 0.00183)
patient_proportion2 <- c(0.0834, 0.0863, 0.152, 0.481, 0.197)

patient_prop <- c(patient_proportion1, patient_proportion2)
dose_prop <- c(1:5, 1:5)
scenario_prop <- c(rep("Scenario 1", 5), rep("Scenario 2", 5))
patient_data <- data.frame(patient_prop, dose_prop, scenario_prop)

patient_plot <- ggplot(data = patient_data, aes(x = dose_prop, y = patient_prop, fill = scenario_prop)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_x_continuous(breaks = seq(1, 5, 1), labels = dose) +
  labs(x = "Dose (mg/m^2/day)", y = "Proportion of patients", title = "Proportion of patients in two scenarios") +
  theme(text = element_text(size = 20))

ggsave("patient_prop.png", patient_plot, width = width, height = height)
