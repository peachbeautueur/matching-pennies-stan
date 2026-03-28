data {
  int<lower=1> T;                       // Number of trials.
  array[T] int<lower=0, upper=1> y;     // Agent choices: 1 = Heads, 0 = Tails.
  array[T] int<lower=0, upper=1> opp;   // Opponent choices on each trial.
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0> beta;
}

model {
  real p;
  real vH;
  real vT;
  real p_choose_H;

  // Priors regularize learning and choice consistency while remaining flexible.
  alpha ~ beta(2, 2);
  beta ~ lognormal(0, 1);

  // The model starts from an unbiased belief: the opponent is equally likely
  // to choose Heads or Tails before any evidence has been observed.
  p = 0.5;

  for (t in 1:T) {
    // In matching pennies, the value of each action depends on the current
    // belief about what the opponent will do next.
    vH = 2 * p - 1;
    vT = 1 - 2 * p;

    // The value difference is mapped to a stochastic choice probability.
    // Higher beta makes choices more deterministic.
    p_choose_H = inv_logit(beta * (vH - vT));

    // Observed choice likelihood.
    y[t] ~ bernoulli(p_choose_H);

    // Delta rule update: the new belief moves part of the way from the old
    // belief toward the opponent's observed action.
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

  // Rebuild the latent belief trajectory so that we can generate posterior
  // predictive choices and compute the overall log likelihood.
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
