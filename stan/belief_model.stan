data {
  int<lower=1> T;                       // number of trials
  array[T] int<lower=0, upper=1> y;     // agent choice: 1 = H, 0 = T
  array[T] int<lower=0, upper=1> opp;   // opponent choice: 1 = H, 0 = T
}

parameters {
  real<lower=0, upper=1> alpha;         // learning rate
  real<lower=0> beta;                   // inverse temperature
}

model {
  real p;                               // belief that opponent will choose H
  real vH;
  real vT;
  real p_choose_H;

  alpha ~ beta(2, 2);
  beta ~ lognormal(0, 1);

  p = 0.5;

  for (t in 1:T) {
    vH = 2 * p - 1;
    vT = 1 - 2 * p;

    p_choose_H = inv_logit(beta * (vH - vT));

    y[t] ~ bernoulli(p_choose_H);

    p = p + alpha * (opp[t] - p);
  }
}

generated quantities {
  array[T] int y_rep;
  real log_lik;
  real p;
  real vH;
  real vT;
  real p_choose_H;

  log_lik = 0;
  p = 0.5;

  for (t in 1:T) {
    vH = 2 * p - 1;
    vT = 1 - 2 * p;
    p_choose_H = inv_logit(beta * (vH - vT));

    y_rep[t] = bernoulli_rng(p_choose_H);
    log_lik += bernoulli_lpmf(y[t] | p_choose_H);

    p = p + alpha * (opp[t] - p);
  }
}