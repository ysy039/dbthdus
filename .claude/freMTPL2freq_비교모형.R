# ============================================================
# freMTPL2freq_비교모형.R
#
# CASdatasets 패키지의 freMTPL2freq(프랑스 자동차보험 청구빈도 데이터,
# 677,991행)에 260821_실제데이터 비교모형.R / telematics_비교모형.R /
# dataCar_비교모형.R과 동일한 4개 모형(제안모형/ZIP/ZINB1/ZINB2_lm)을 적용한다.
#
# 반응변수: ClaimNb (계약기간 중 청구 건수). IDpol(계약 식별자, 677,991개
# 전부 고유값)은 정보가 없는 ID 컬럼이라 제외.
# 설명변수: Exposure, VehPower, VehAge, DrivAge, BonusMalus, VehBrand(11개
# 범주), VehGas, Area(6개 범주), Density, Region(22개 범주) 전부 포함.
#
# telematics/dataCar와 마찬가지로 반응변수가 극단적 영과잉(0 비율 96.3%,
# mean=0.039, var=0.043 -> 거의 순수 Poisson에 가까움)이라, 계산량을 고려해
# n=10,000으로 무작위 서브샘플링하고 단일 80/20 분할 1회로 평가한다.
# (telematics 44개 공변량에서 ZINB2_lm이 유사분리로 발산했던 것과 비슷하게,
#  Region 22개+VehBrand 11개 원-핫 인코딩으로 공변량이 많아 유사한 현상이
#  재현될 수 있음 - 발산하면 그대로 결과에 남긴다는 앞선 방침을 그대로 따른다.)
# ============================================================

library(xgboost)
library(pscl)

# ============================================================
# 1. 제안모형: ZINB2 boosting (mu, p 분리)
# ============================================================
zinb2_nll_vec <- function(y, mu, p, alpha) {
  r <- 1 / alpha
  nb0 <- dnbinom(0, size = r, mu = mu)
  is_zero <- y == 0
  nll <- numeric(length(y))
  prob_zero <- p[is_zero] + (1 - p[is_zero]) * nb0[is_zero]
  nll[is_zero] <- -log(pmax(prob_zero, .Machine$double.eps))
  if (any(!is_zero)) {
    nll[!is_zero] <-
      -log(pmax(1 - p[!is_zero], .Machine$double.eps)) -
      dnbinom(y[!is_zero], size = r, mu = mu[!is_zero], log = TRUE)
  }
  nll
}

optimize_alpha_zinb2 <- function(y, mu, p) {
  obj_log_alpha <- function(log_alpha) {
    alpha <- exp(log_alpha)
    if (!is.finite(alpha)) return(.Machine$double.xmax)
    loss <- sum(zinb2_nll_vec(y = y, mu = mu, p = p, alpha = alpha))
    if (!is.finite(loss)) return(.Machine$double.xmax)
    loss
  }
  res <- optimize(f = obj_log_alpha, interval = log(c(0.1, 5)))
  exp(as.numeric(res$minimum))
}

zinb_objective_nb <- function(alpha, p_fixed) {
  force(alpha)
  function(preds, dtrain) {
    y <- xgboost::getinfo(dtrain, "label")
    mu <- exp(preds)
    A <- 1 + alpha * mu; B <- exp(log(A) / alpha); D <- p_fixed * B + (1 - p_fixed)
    is_zero <- y == 0
    grad <- numeric(length(y)); hess <- numeric(length(y))
    grad[is_zero] <- (1 - p_fixed[is_zero]) * mu[is_zero] / (A[is_zero] * D[is_zero])
    hess[is_zero] <- (1 - p_fixed[is_zero]) * mu[is_zero] *
      (D[is_zero] - mu[is_zero] * p_fixed[is_zero] * B[is_zero]) / (A[is_zero]^2 * D[is_zero]^2)
    grad[!is_zero] <- -y[!is_zero] + ((alpha * y[!is_zero] + 1) * mu[!is_zero]) / A[!is_zero]
    hess[!is_zero] <- ((alpha * y[!is_zero] + 1) * mu[!is_zero]) / A[!is_zero]^2
    list(grad = grad, hess = hess)
  }
}

zinb_objective_logit <- function(alpha, mu_fixed) {
  force(alpha)
  function(preds, dtrain) {
    y <- xgboost::getinfo(dtrain, "label")
    p <- plogis(preds); mu <- mu_fixed
    A <- 1 + alpha * mu; B <- exp(log(A) / alpha); D <- p * B + (1 - p)
    is_zero <- y == 0
    grad <- numeric(length(y)); hess <- numeric(length(y)); u <- p * (1 - p)
    grad[is_zero] <- -u[is_zero] * (B[is_zero] - 1) / D[is_zero]
    hess[is_zero] <- (B[is_zero] - 1) * u[is_zero] *
      ((2 * p[is_zero] - 1) * D[is_zero] + u[is_zero] * (B[is_zero] - 1)) / D[is_zero]^2
    grad[!is_zero] <- p[!is_zero]; hess[!is_zero] <- u[!is_zero]
    list(grad = grad, hess = hess)
  }
}

zinb2_boost <- function(data, y, t_max = 500, lr_nb = 0.01, lr_logit = 0.02,
                         alpha = 1, conv_tol = 1e-2, max_alpha_iter = 20,
                         verbose = FALSE, max_depth_nb = 1, max_depth_logit = 1) {
  x_mat <- as.matrix(data); n <- nrow(x_mat)
  dtrain <- xgboost::xgb.DMatrix(data = x_mat, label = y)
  dpred  <- xgboost::xgb.DMatrix(data = x_mat)
  alpha_cur <- alpha; mu_prev <- NULL; p_prev <- NULL
  final_models_nb <- NULL; final_models_logit <- NULL
  final_mu_hat <- NULL; final_p_hat <- NULL; final_alpha_hat <- NULL
  converged <- FALSE; used_iter <- 0

  for (k in seq_len(max_alpha_iter)) {
    f_nb <- rep(0, n); f_logit <- rep(0, n)
    mu_hat <- exp(f_nb); p_hat <- plogis(f_logit)
    all_models_nb <- vector("list", t_max); all_models_logit <- vector("list", t_max)

    for (t in seq_len(t_max)) {
      xgboost::setinfo(dtrain, "base_margin", f_nb)
      obj_nb <- zinb_objective_nb(alpha = alpha_cur, p_fixed = p_hat)
      bst_nb <- xgboost::xgb.train(
        params = list(booster = "gbtree", max_depth = max_depth_nb, eta = 1,
                      base_score = 0, disable_default_eval_metric = TRUE),
        data = dtrain, nrounds = 1, obj = obj_nb, verbose = 0)
      inc_nb <- predict(bst_nb, dpred, outputmargin = TRUE)
      f_nb <- f_nb + lr_nb * inc_nb; mu_hat <- exp(f_nb)
      all_models_nb[[t]] <- bst_nb

      xgboost::setinfo(dtrain, "base_margin", f_logit)
      obj_logit <- zinb_objective_logit(alpha = alpha_cur, mu_fixed = mu_hat)
      bst_logit <- xgboost::xgb.train(
        params = list(booster = "gbtree", max_depth = max_depth_logit, eta = 1,
                      base_score = 0, disable_default_eval_metric = TRUE),
        data = dtrain, nrounds = 1, obj = obj_logit, verbose = 0)
      inc_logit <- predict(bst_logit, dpred, outputmargin = TRUE)
      f_logit <- f_logit + lr_logit * inc_logit; p_hat <- plogis(f_logit)
      all_models_logit[[t]] <- bst_logit
    }

    alpha_new <- optimize_alpha_zinb2(y = y, mu = mu_hat, p = p_hat)
    delta_mu <- if (is.null(mu_prev)) Inf else max(abs(mu_hat - mu_prev) / abs(mu_prev))
    delta_p  <- if (is.null(p_prev))  Inf else max(abs(p_hat - p_prev))

    final_models_nb <- all_models_nb; final_models_logit <- all_models_logit
    final_mu_hat <- mu_hat; final_p_hat <- p_hat; final_alpha_hat <- alpha_new; used_iter <- k

    if (verbose) {
      cat(sprintf("[proposed k=%02d] alpha_prev=%.4f -> alpha_new=%.4f | max_rel_delta_mu=%.6f | max_abs_delta_p=%.6f\n",
                  k, alpha_cur, alpha_new, delta_mu, delta_p))
      flush.console()
    }

    if (!is.null(mu_prev) && delta_mu < conv_tol && delta_p < conv_tol) { converged <- TRUE; break }
    mu_prev <- mu_hat; p_prev <- p_hat; alpha_cur <- alpha_new
  }

  list(mu_hat = final_mu_hat, p_hat = final_p_hat, alpha_hat = final_alpha_hat,
       models_nb = final_models_nb, models_logit = final_models_logit,
       lr_nb = lr_nb, lr_logit = lr_logit, converged = converged, used_iter = used_iter)
}

predict_zinb2_boost <- function(fit, newx) {
  newx <- as.matrix(newx)
  dnew <- xgboost::xgb.DMatrix(data = newx)
  f_nb <- rep(0, nrow(newx))
  for (m in fit$models_nb) f_nb <- f_nb + fit$lr_nb * predict(m, dnew, outputmargin = TRUE)
  f_logit <- rep(0, nrow(newx))
  for (m in fit$models_logit) f_logit <- f_logit + fit$lr_logit * predict(m, dnew, outputmargin = TRUE)
  list(mu_hat = exp(f_nb), p_hat = plogis(f_logit))
}

# ============================================================
# 2. ZIP 비교모형: mu, p 분리 boosting, 분산=mu인 Poisson 기반
# ============================================================
zip_nll_vec <- function(y, mu, p, include_constant = TRUE) {
  eps <- .Machine$double.eps
  mu <- pmax(mu, 1e-12); p <- pmin(pmax(p, 1e-8), 1 - 1e-8)
  is_zero <- y == 0
  nll <- numeric(length(y))
  prob_zero <- p[is_zero] + (1 - p[is_zero]) * exp(-mu[is_zero])
  nll[is_zero] <- -log(pmax(prob_zero, eps))
  if (any(!is_zero)) {
    idx <- !is_zero
    nll[idx] <- -log(pmax(1 - p[idx], eps)) + mu[idx] - y[idx] * log(mu[idx])
    if (include_constant) nll[idx] <- nll[idx] + lgamma(y[idx] + 1)
  }
  nll
}

zip_objective_mu <- function(p_fixed) {
  force(p_fixed)
  function(preds, dtrain) {
    y <- xgboost::getinfo(dtrain, "label")
    mu <- exp(preds); p <- p_fixed
    is_zero <- y == 0
    grad <- numeric(length(y)); hess <- numeric(length(y))
    if (any(is_zero)) {
      idx <- is_zero
      pois_zero <- exp(-mu[idx])
      q_zero <- pmax(p[idx] + (1 - p[idx]) * pois_zero, .Machine$double.eps)
      grad[idx] <- (1 - p[idx]) * mu[idx] * pois_zero / q_zero
      hess[idx] <- grad[idx] * (1 - mu[idx] + grad[idx])
    }
    if (any(!is_zero)) {
      idx <- !is_zero
      grad[idx] <- mu[idx] - y[idx]
      hess[idx] <- mu[idx]
    }
    list(grad = grad, hess = hess)
  }
}

zip_objective_logit <- function(mu_fixed) {
  force(mu_fixed)
  function(preds, dtrain) {
    y <- xgboost::getinfo(dtrain, "label")
    p <- plogis(preds); mu <- mu_fixed
    is_zero <- y == 0
    grad <- numeric(length(y)); hess <- numeric(length(y)); u <- p * (1 - p)
    if (any(is_zero)) {
      idx <- is_zero
      pois_zero <- exp(-mu[idx]); c_zero <- 1 - pois_zero
      q_zero <- pmax(p[idx] + (1 - p[idx]) * pois_zero, .Machine$double.eps)
      grad[idx] <- -u[idx] * c_zero / q_zero
      hess[idx] <- c_zero * u[idx] * ((2 * p[idx] - 1) * q_zero + u[idx] * c_zero) / q_zero^2
    }
    if (any(!is_zero)) {
      idx <- !is_zero
      grad[idx] <- p[idx]; hess[idx] <- u[idx]
    }
    list(grad = grad, hess = hess)
  }
}

zip_boost <- function(data, y, t_max = 500, lr_nb = 0.01, lr_logit = 0.02,
                       verbose = FALSE, max_depth_nb = 1, max_depth_logit = 1,
                       init_mu = 1, init_p = 0.5) {
  x_mat <- as.matrix(data); n <- nrow(x_mat)
  dtrain <- xgboost::xgb.DMatrix(data = x_mat, label = y)
  dpred  <- xgboost::xgb.DMatrix(data = x_mat)

  f_mu <- rep(log(init_mu), n)
  f_logit <- rep(qlogis(init_p), n)
  mu_hat <- exp(f_mu); p_hat <- plogis(f_logit)

  models_mu <- vector("list", t_max); models_logit <- vector("list", t_max)

  for (t in seq_len(t_max)) {
    xgboost::setinfo(dtrain, "base_margin", f_mu)
    obj_mu <- zip_objective_mu(p_fixed = p_hat)
    bst_mu <- xgboost::xgb.train(
      params = list(booster = "gbtree", max_depth = max_depth_nb, eta = 1,
                    base_score = 0, disable_default_eval_metric = TRUE),
      data = dtrain, nrounds = 1, obj = obj_mu, verbose = 0)
    f_mu <- f_mu + lr_nb * predict(bst_mu, dpred, outputmargin = TRUE)
    mu_hat <- exp(f_mu)
    models_mu[[t]] <- bst_mu

    xgboost::setinfo(dtrain, "base_margin", f_logit)
    obj_logit <- zip_objective_logit(mu_fixed = mu_hat)
    bst_logit <- xgboost::xgb.train(
      params = list(booster = "gbtree", max_depth = max_depth_logit, eta = 1,
                    base_score = 0, disable_default_eval_metric = TRUE),
      data = dtrain, nrounds = 1, obj = obj_logit, verbose = 0)
    f_logit <- f_logit + lr_logit * predict(bst_logit, dpred, outputmargin = TRUE)
    p_hat <- plogis(f_logit)
    models_logit[[t]] <- bst_logit
  }

  list(mu_hat = mu_hat, p_hat = p_hat, models_mu = models_mu, models_logit = models_logit,
       lr_nb = lr_nb, lr_logit = lr_logit, init_mu = init_mu, init_p = init_p)
}

predict_zip_boost <- function(fit, newx) {
  newx <- as.matrix(newx)
  dnew <- xgboost::xgb.DMatrix(data = newx)
  f_mu <- rep(log(fit$init_mu), nrow(newx))
  for (m in fit$models_mu) f_mu <- f_mu + fit$lr_nb * predict(m, dnew, outputmargin = TRUE)
  f_logit <- rep(qlogis(fit$init_p), nrow(newx))
  for (m in fit$models_logit) f_logit <- f_logit + fit$lr_logit * predict(m, dnew, outputmargin = TRUE)
  list(mu_hat = exp(f_mu), p_hat = plogis(f_logit))
}

# ============================================================
# 3. ZINB1 비교모형: mu만 boosting, p는 mu에서 유도 (p = 1/(1+mu^gam))
# ============================================================
zinb_mu_nll_vec <- function(y, mu, gam, alpha) {
  mu <- pmax(mu, 1e-12)
  p <- plogis(-gam * log(mu))
  size <- 1 / alpha
  is_zero <- y == 0
  nll <- numeric(length(y))
  nb0 <- dnbinom(0, size = size, mu = mu)
  prob_zero <- p[is_zero] + (1 - p[is_zero]) * nb0[is_zero]
  nll[is_zero] <- -log(pmax(prob_zero, .Machine$double.eps))
  if (any(!is_zero)) {
    nll[!is_zero] <- -log(pmax(1 - p[!is_zero], .Machine$double.eps)) -
      dnbinom(y[!is_zero], size = size, mu = mu[!is_zero], log = TRUE)
  }
  nll
}

zinb_objective_mu <- function(gam, alpha) {
  force(gam); force(alpha)
  function(preds, dtrain) {
    y <- xgboost::getinfo(dtrain, "label")
    ln_mu <- preds; mu <- exp(ln_mu)
    A <- 1 + alpha * mu
    w <- plogis(gam * ln_mu)
    is_zero <- y == 0
    grad <- numeric(length(y)); hess <- numeric(length(y))

    log_C <- gam * ln_mu - log(A) / alpha
    z <- plogis(log_C)
    a <- gam - mu / A

    grad[is_zero] <- gam * w[is_zero] - z[is_zero] * a[is_zero]
    hess[is_zero] <- gam^2 * w[is_zero] * (1 - w[is_zero]) -
      z[is_zero] * (1 - z[is_zero]) * a[is_zero]^2 +
      z[is_zero] * mu[is_zero] / A[is_zero]^2

    grad[!is_zero] <- -gam + gam * w[!is_zero] - y[!is_zero] +
      (alpha * y[!is_zero] + 1) * mu[!is_zero] / A[!is_zero]
    hess[!is_zero] <- gam^2 * w[!is_zero] * (1 - w[!is_zero]) +
      (alpha * y[!is_zero] + 1) * mu[!is_zero] / A[!is_zero]^2

    list(grad = grad, hess = hess)
  }
}

optimize_alpha_zinb_mu <- function(y, mu, gam, alpha_start, alpha_lower = 0.1, alpha_upper = 5) {
  objective_log_alpha <- function(log_alpha) {
    alpha_candidate <- exp(log_alpha)
    loss <- sum(zinb_mu_nll_vec(y = y, mu = mu, gam = gam, alpha = alpha_candidate))
    if (!is.finite(loss)) return(.Machine$double.xmax)
    loss
  }
  result <- optim(par = log(alpha_start), fn = objective_log_alpha, method = "L-BFGS-B",
                   lower = log(alpha_lower), upper = log(alpha_upper))
  exp(as.numeric(result$par))
}

zinb_mu_boost <- function(data, y, t_max = 500, lr_mu = 0.01, gam = 1, alpha = 1,
                           conv_tol = 1e-2, max_alpha_iter = 20, verbose = FALSE,
                           max_depth_mu = 1, alpha_lower = 0.1, alpha_upper = 5) {
  x_mat <- as.matrix(data); n <- nrow(x_mat)
  dtrain <- xgboost::xgb.DMatrix(data = x_mat, label = y)
  dpred  <- xgboost::xgb.DMatrix(data = x_mat)

  alpha_cur <- alpha; mu_prev <- NULL
  final_models_mu <- NULL; final_mu_hat <- NULL; final_alpha_hat <- NULL
  converged <- FALSE; used_iter <- 0

  for (k in seq_len(max_alpha_iter)) {
    f_mu <- rep(0, n)
    mu_hat <- exp(f_mu)
    all_models_mu <- vector("list", t_max)

    for (t in seq_len(t_max)) {
      xgboost::setinfo(dtrain, "base_margin", f_mu)
      obj_mu <- zinb_objective_mu(gam = gam, alpha = alpha_cur)
      bst_mu <- xgboost::xgb.train(
        params = list(booster = "gbtree", max_depth = max_depth_mu, eta = 1,
                      base_score = 0, disable_default_eval_metric = TRUE),
        data = dtrain, nrounds = 1, obj = obj_mu, verbose = 0)
      f_mu <- f_mu + lr_mu * predict(bst_mu, dpred, outputmargin = TRUE)
      mu_hat <- exp(f_mu)
      all_models_mu[[t]] <- bst_mu
    }

    alpha_new <- optimize_alpha_zinb_mu(y = y, mu = mu_hat, gam = gam, alpha_start = alpha_cur,
                                         alpha_lower = alpha_lower, alpha_upper = alpha_upper)
    delta_mu <- if (is.null(mu_prev)) Inf else max(abs(mu_hat - mu_prev) / abs(mu_prev))

    final_models_mu <- all_models_mu; final_mu_hat <- mu_hat
    final_alpha_hat <- alpha_new; used_iter <- k

    if (verbose) {
      cat(sprintf("[ZINB1 k=%02d] alpha_prev=%.4f -> alpha_new=%.4f | max_rel_delta_mu=%.6f\n",
                  k, alpha_cur, alpha_new, delta_mu))
      flush.console()
    }

    if (!is.null(mu_prev) && delta_mu < conv_tol) { converged <- TRUE; break }
    mu_prev <- mu_hat; alpha_cur <- alpha_new
  }

  list(mu_hat = final_mu_hat, alpha_hat = final_alpha_hat, models_mu = final_models_mu,
       lr_mu = lr_mu, gam = gam, converged = converged, used_iter = used_iter)
}

predict_zinb_mu_boost <- function(fit, newx) {
  newx <- as.matrix(newx)
  dnew <- xgboost::xgb.DMatrix(data = newx)
  f_mu <- rep(0, nrow(newx))
  for (m in fit$models_mu) f_mu <- f_mu + fit$lr_mu * predict(m, dnew, outputmargin = TRUE)
  mu_hat <- exp(f_mu)
  p_hat <- plogis(-fit$gam * f_mu)
  list(mu_hat = mu_hat, p_hat = p_hat)
}

# ============================================================
# 4. ZINB2 선형회귀 비교모형: pscl::zeroinfl(dist="negbin")
#    mu, p를 각각 별도의 "선형" 예측식(전체 공변량)으로 적합
# ============================================================
fit_zinb2_regression <- function(data, y) {
  x_mat <- as.matrix(data)

  # model.matrix(~ . - 1)로 원-핫 인코딩하면 "-1"(절편 제거) 특성상 R이 맨 처음
  # 등장하는 범주형 변수 하나에만 전체 수준(레벨)을 다 부여해서 절편 역할을
  # 대신하게 만든다. 이 컬럼들의 합이 상수(=1)라서, 절편이 있는
  # pscl::zeroinfl에 그대로 넣으면 설계행렬이 특이(rank-deficient)해져
  # "non-finite value supplied by optim" 에러가 난다. -> 절편을 명시적으로
  # 추가해 랭크를 확인하고 중복 컬럼을 자동 제거한 뒤, 절편이 있는 표준적인
  # 형태로 적합한다.
  x_with_int <- cbind(Intercept = 1, x_mat)
  qr_x <- qr(x_with_int)
  r <- qr_x$rank
  keep_idx <- sort(qr_x$pivot[seq_len(r)])
  keep_idx <- keep_idx[keep_idx != 1] - 1
  if (length(keep_idx) < ncol(x_mat)) {
    dropped <- setdiff(seq_len(ncol(x_mat)), keep_idx)
    cat(sprintf("  [ZINB2_lm] 설계행렬 rank-deficient -> %d개 컬럼 제거: %s\n",
                length(dropped), paste(colnames(x_mat)[dropped], collapse = ", ")))
  }
  x_mat_fixed <- x_mat[, keep_idx, drop = FALSE]

  x_df <- as.data.frame(x_mat_fixed)
  names(x_df) <- paste0("x", seq_len(ncol(x_df)))
  reg_data <- data.frame(y = y, x_df)
  xvars <- paste(names(x_df), collapse = " + ")
  fml <- stats::as.formula(sprintf("y ~ %s | %s", xvars, xvars))

  try_fit <- function(method) {
    tryCatch(
      suppressWarnings(pscl::zeroinfl(
        fml, data = reg_data, dist = "negbin", link = "logit",
        control = pscl::zeroinfl.control(method = method, maxit = 10000, trace = FALSE)
      )),
      error = function(e) NULL
    )
  }

  fit <- try_fit("BFGS")
  if (is.null(fit)) fit <- try_fit("Nelder-Mead")

  list(fit = fit, keep_idx = keep_idx)
}

predict_zinb2_regression <- function(fit_obj, newx) {
  n_new <- nrow(newx)
  if (is.null(fit_obj$fit)) {
    return(list(mu_hat = rep(NA_real_, n_new), p_hat = rep(NA_real_, n_new)))
  }
  newx_mat <- as.matrix(newx)[, fit_obj$keep_idx, drop = FALSE]
  newx_df <- as.data.frame(newx_mat)
  names(newx_df) <- paste0("x", seq_len(ncol(newx_df)))
  mu_hat <- as.numeric(predict(fit_obj$fit, newdata = newx_df, type = "count"))
  p_hat  <- as.numeric(predict(fit_obj$fit, newdata = newx_df, type = "zero"))
  list(mu_hat = mu_hat, p_hat = p_hat)
}

# ============================================================
# 5. 실제 데이터에 네 모형 모두 학습 + 평가 (train/test 1회 분할)
# ============================================================
hp <- list(t_max = 500, lr_nb = 0.01, lr_logit = 0.02, lr_mu = 0.01,
           conv_tol = 1e-2, max_alpha_iter = 20,
           max_depth_nb = 1, max_depth_logit = 1, max_depth_mu = 1, gam = 1)

run_real_data_compare <- function(X, y, dataset_name, train_frac = 0.8, seed = 1) {
  n <- nrow(X)

  set.seed(seed)
  train_idx <- sample(seq_len(n), size = floor(train_frac * n))
  test_idx  <- setdiff(seq_len(n), train_idx)

  x_train <- X[train_idx, , drop = FALSE]; y_train <- y[train_idx]
  x_test  <- X[test_idx, , drop = FALSE];  y_test  <- y[test_idx]

  cat(sprintf("\n########## %s (n=%d, train=%d, test=%d) ##########\n",
              dataset_name, n, length(train_idx), length(test_idx)))
  cat(sprintf("mean(y)=%.4f, var(y)=%.4f, y=0 ratio=%.4f\n", mean(y), var(y), mean(y == 0)))

  # ---- 1) 제안모형 ----
  fit_proposed <- zinb2_boost(
    data = x_train, y = y_train, t_max = hp$t_max, lr_nb = hp$lr_nb, lr_logit = hp$lr_logit,
    alpha = 1, conv_tol = hp$conv_tol, max_alpha_iter = hp$max_alpha_iter,
    verbose = FALSE, max_depth_nb = hp$max_depth_nb, max_depth_logit = hp$max_depth_logit
  )
  pred_proposed <- predict_zinb2_boost(fit_proposed, x_test)
  alpha_proposed <- optimize_alpha_zinb2(y_test, pred_proposed$mu_hat, pred_proposed$p_hat)

  # ---- 2) ZIP 비교모형 ----
  fit_zip <- zip_boost(
    data = x_train, y = y_train, t_max = hp$t_max, lr_nb = hp$lr_nb, lr_logit = hp$lr_logit,
    verbose = FALSE, max_depth_nb = hp$max_depth_nb, max_depth_logit = hp$max_depth_logit
  )
  pred_zip <- predict_zip_boost(fit_zip, x_test)

  # ---- 3) ZINB1 비교모형 ----
  fit_reg <- zinb_mu_boost(
    data = x_train, y = y_train, t_max = hp$t_max, lr_mu = hp$lr_mu, gam = hp$gam,
    alpha = 1, conv_tol = hp$conv_tol, max_alpha_iter = hp$max_alpha_iter,
    verbose = FALSE, max_depth_mu = hp$max_depth_mu
  )
  pred_reg <- predict_zinb_mu_boost(fit_reg, x_test)
  alpha_reg <- optimize_alpha_zinb2(y_test, pred_reg$mu_hat, pred_reg$p_hat)

  # ---- 4) ZINB2 선형회귀 비교모형 ----
  fit_lm <- fit_zinb2_regression(data = x_train, y = y_train)
  pred_lm <- predict_zinb2_regression(fit_lm, x_test)
  # pscl은 Var=mu+mu^2/theta로 parametrize -> 우리 alpha(=1/theta)로 환산
  alpha_lm <- if (!is.null(fit_lm$fit) && !is.null(fit_lm$fit$theta)) 1 / fit_lm$fit$theta else NA_real_

  y_hat_proposed <- (1 - pred_proposed$p_hat) * pred_proposed$mu_hat
  y_hat_zip      <- (1 - pred_zip$p_hat)      * pred_zip$mu_hat
  y_hat_reg      <- (1 - pred_reg$p_hat)      * pred_reg$mu_hat
  y_hat_lm       <- (1 - pred_lm$p_hat)       * pred_lm$mu_hat

  res <- data.frame(
    dataset = dataset_name,
    model = c("proposed", "zip", "ZINB1", "ZINB2_lm"),
    converged = c(fit_proposed$converged, NA, fit_reg$converged, !is.null(fit_lm$fit)),
    used_iter = c(fit_proposed$used_iter, NA, fit_reg$used_iter, NA),
    MAE_y = c(
      mean(abs(y_test - y_hat_proposed)),
      mean(abs(y_test - y_hat_zip)),
      mean(abs(y_test - y_hat_reg)),
      mean(abs(y_test - y_hat_lm), na.rm = TRUE)
    ),
    alpha_hat = c(alpha_proposed, NA, alpha_reg, alpha_lm),
    mean_mu_hat = c(mean(pred_proposed$mu_hat), mean(pred_zip$mu_hat), mean(pred_reg$mu_hat), mean(pred_lm$mu_hat, na.rm = TRUE)),
    mean_p_hat  = c(mean(pred_proposed$p_hat),  mean(pred_zip$p_hat),  mean(pred_reg$p_hat),  mean(pred_lm$p_hat, na.rm = TRUE))
  )

  cat("\n")
  print(res, row.names = FALSE, digits = 4)

  res
}

# ============================================================
# 6. freMTPL2freq 데이터 불러오기 + 전처리
#    - 반응변수: ClaimNb (청구 건수)
#    - 설명변수: Exposure, VehPower, VehAge, DrivAge, BonusMalus,
#      VehBrand(11개 범주), VehGas, Area(6개 범주), Density, Region(22개 범주)
#    - 제외: IDpol (계약 식별자, 정보 없는 ID)
#    - n=10,000으로 무작위 서브샘플링 (seed=1), telematics/dataCar와 동일한 방식
# ============================================================
library(CASdatasets)
data(freMTPL2freq)

set.seed(1)
sub_idx <- sample(seq_len(nrow(freMTPL2freq)), size = 50000)
fre_df <- freMTPL2freq[sub_idx, ]

y_fre <- fre_df$ClaimNb

x_vars <- setdiff(names(fre_df), c("IDpol", "ClaimNb"))
X_fre <- model.matrix(~ . - 1, data = fre_df[, x_vars])

res_fre <- run_real_data_compare(X_fre, y_fre, "freMTPL2freq (ClaimNb)", train_frac = 0.8, seed = 1)

# ============================================================
# 7. 결과 저장
# ============================================================
write.csv(res_fre, "freMTPL2freq_비교모형_결과.csv", row.names = FALSE)
cat("\nSaved freMTPL2freq_비교모형_결과.csv\n")
