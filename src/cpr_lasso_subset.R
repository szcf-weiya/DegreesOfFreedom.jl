library(mvtnorm)
library(glmnet)
library(leaps)
n = 20; p = 10; N = 100
set.seed(20210319)
#set.seed(999)
Sigma = matrix(0, p, p)
Sigma[1:4, 1:4] = 0.75
Sigma[5:10, 5:10] = 0.9
diag(Sigma) = 1
X = rmvnorm(n, rep(0, p), Sigma)
beta.true = c(1:5, rep(0, 5))
y = X %*% beta.true + rnorm(n)
lasso.fit = glmnet(X, y, alpha = 1, lambda = 10^seq(0.6, -2, length.out = 10))
lasso.beta = as.matrix(lasso.fit$beta)
lasso.beta
apply(lasso.beta, 2, function(x) sum(x != 0))
# s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
# 1  4  4  5  6  6  6  7  8 10
lambda.grid = 10^seq(1, -2, length.out = 10)
num_nonzeros = matrix(0, N, p*2)
Y = matrix(0, nrow = n, ncol = N)
Yhat = vector("list", 10*2)
for (k in 1:20) Yhat[[k]] = matrix(0, nrow = n, ncol = N)
for (i in 1:100) {
  Y[,i] = X %*% beta.true + rnorm(n)
  lasso.fit = glmnet(X, Y[,i], alpha = 1, lambda = lambda.grid, intercept = FALSE)
  lasso.beta = as.matrix(lasso.fit$beta) # one column is for one lambda
  num_nonzeros[i, 1:p] = apply(lasso.beta, 2, function(x) sum(x != 0))
  bs.fit = regsubsets(X, Y[, i], intercept = FALSE, nvmax = 10)
  bs.which = summary(bs.fit)$which # one row is for one subset
  num_nonzeros[i, (p+1):(2*p)] = apply(bs.which, 1, sum)
  for (k in 1:10) {
    Yhat[[k]][, i] = X %*% lasso.beta[, k]
    Yhat[[10+k]][, i] = as.matrix(X[, bs.which[k, ]]) %*% as.matrix(coef(bs.fit, k))
  }
}
nns = apply(num_nonzeros, 2, mean)
dfs = numeric(20)
for (k in 1:20) {
  dfs[k] = sum(sapply(1:n, function(i) cov(Y[i, ], Yhat[[k]][i, ])))
}

# plot(c(0,nns[1:10]), c(0,dfs[1:10]), xlab = "Average # of nonzero coefficients", ylab = "DoF", pch=1, col = "blue", type="o", xlim = c(0, 10), ylim = c(0, 10))
# lines(c(0,nns[11:20]), c(0,dfs[11:20]), type = "o", pch = 2, col = "red")
# abline(a=0, b=1, lty=2)
# legend("bottomright", pch = c(1, 2), col = c("blue", "red"), legend = c("lasso", "best subset"))
