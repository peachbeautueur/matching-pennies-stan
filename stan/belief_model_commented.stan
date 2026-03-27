data {
  int<lower=1> T;                       // Number of trials in the observed sequence.
  array[T] int<lower=0, upper=1> y;     // Agent choices: 1 = Heads, 0 = Tails.
  array[T] int<lower=0, upper=1> opp;   // Opponent choices: 1 = Heads, 0 = Tails.
}

parameters {
  real<lower=0, upper=1> alpha;         // Learning rate for belief updating.
  real<lower=0> beta;                   // Inverse temperature for softmax choice.
}

model {
  real p;                               // Current belief that opponent will choose Heads.
  real vH;                              // Subjective value of choosing Heads.
  real vT;                              // Subjective value of choosing Tails.
  real p_choose_H;                      // Choice probability of Heads after softmax.

  alpha ~ beta(2, 2);                   // Prior favoring interior learning-rate values.
  beta ~ lognormal(0, 1);               // Positive prior for choice sensitivity.

  p = 0.5;                              // Initial belief: opponent is equally likely to play H/T.

  for (t in 1:T) {
    vH = 2 * p - 1;                     // Expected payoff of Heads in matching pennies.
    vT = 1 - 2 * p;                     // Expected payoff of Tails.

    p_choose_H = inv_logit(beta * (vH - vT));  // Softmax on the value difference.

    y[t] ~ bernoulli(p_choose_H);       // Likelihood for the observed choice.

    p = p + alpha * (opp[t] - p);       // Delta-rule update after observing the opponent.
  }
}

generated quantities {
  array[T] int y_rep;                   // Posterior predictive replicated choices.
  real log_lik;                         // Pointwise log-likelihood summed over trials.
  real p;                               // Re-created latent belief for posterior simulation.
  real vH;                              // Re-created value of Heads.
  real vT;                              // Re-created value of Tails.
  real p_choose_H;                      // Re-created choice probability for Heads.

  log_lik = 0;                          // Initialize the accumulated log-likelihood.
  p = 0.5;                              // Use the same initial condition as in the model block.

  for (t in 1:T) {
    vH = 2 * p - 1;                     // Recompute trial-wise action value of Heads.
    vT = 1 - 2 * p;                     // Recompute trial-wise action value of Tails.
    p_choose_H = inv_logit(beta * (vH - vT));  // Recompute trial-wise choice probability.

    y_rep[t] = bernoulli_rng(p_choose_H);      // Simulate a replicated choice.
    log_lik += bernoulli_lpmf(y[t] | p_choose_H);  // Add this trial's log-likelihood.

    p = p + alpha * (opp[t] - p);       // Advance the latent belief to the next trial.
  }
}
