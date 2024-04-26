library(mvtnorm)
library(glmnet)
library(leaps)
lasso_vs_subset = function(N = 500, n = 20, p = 10, nlam = 10, seed = 20210319) {
  set.seed(seed)
  Sigma = matrix(0, p, p)
  Sigma[1:4, 1:4] = 0.75
  Sigma[5:10, 5:10] = 0.9
  diag(Sigma) = 1
  X = rmvnorm(n, rep(0, p), Sigma)
  beta.true = c(1:5, rep(0, 5))
  y = X %*% beta.true + rnorm(n)
  
  lambda.grid = 10^seq(1, -2, length.out = nlam)
  num_nonzeros = matrix(0, N, p*2)
  Y = matrix(0, nrow = n, ncol = N)
  Yhat.subset = vector("list", p)
  Yhat.lasso = vector("list", nlam)
  n.non.subset = matrix(0, N, p)
  n.non.lasso = matrix(0, N, nlam)
  for (k in 1:p) Yhat.subset[[k]] = matrix(0, nrow = n, ncol = N)
  for (k in 1:nlam) Yhat.lasso[[k]] = matrix(0, nrow = n, ncol = N)
  for (i in 1:N) {
    Y[,i] = X %*% beta.true + rnorm(n)
    lasso.fit = glmnet(X, Y[,i], alpha = 1, lambda = lambda.grid, intercept = FALSE)
    lasso.beta = as.matrix(lasso.fit$beta) # one column is for one lambda
    n.non.lasso[i, ] = apply(lasso.beta, 2, function(x) sum(x != 0))
    bs.fit = regsubsets(X, Y[, i], intercept = FALSE, nvmax = 10)
    bs.which = summary(bs.fit)$which # one row is for one subset
    n.non.subset[i, ] = apply(bs.which, 1, sum)
    for (k in 1:p) {
      Yhat.subset[[k]][, i] = as.matrix(X[, bs.which[k, ]]) %*% as.matrix(coef(bs.fit, k))
    }
    for (k in 1:nlam) {
      Yhat.lasso[[k]][, i] = X %*% lasso.beta[, k]
    }
  }
  avg.n.non.lasso = apply(n.non.lasso, 2, mean)
  avg.n.non.subset = apply(n.non.subset, 2, mean)
  df.lasso = numeric(nlam)
  df.subset = numeric(p)
  for (k in 1:p) df.subset[k] = sum(sapply(1:n, function(i) cov(Y[i, ], Yhat.subset[[k]][i, ])))
  for (k in 1:nlam) df.lasso[k] = sum(sapply(1:n, function(i) cov(Y[i, ], Yhat.lasso[[k]][i, ])))
  list(n.lasso = avg.n.non.lasso, n.subset = avg.n.non.subset, df.lasso = df.lasso, df.subset = df.subset)
}

plot_lasso_vs_subset = function(n.lasso, n.subset, df.lasso, df.subset) {
  plot(c(0, n.lasso), c(0, df.lasso), xlab = "Average # of nonzero coefficients", ylab = "DoF", pch=1, col = "blue", type="o", xlim = c(0, 10), ylim = c(0, 10))
  lines(c(0, n.subset), c(0, df.subset), type = "o", pch = 2, col = "red")
  abline(a = 0, b = 1, lty = 2)
  legend("bottomright", pch = c(1, 2), col = c("blue", "red"), legend = c("lasso", "best subset"))
}
