data {
  int<lower=1> N;                  //number of data points
  int<lower=1> J;                  //number of subjects
  int<lower=1> K;                  //number of items
  int<lower=1,upper=4> say[N];     //outcome: "definitely would"/"probably would"/"probably wouldn't"/"definitely wouldn't"
  int<lower=1, upper=J> subj[N];   //subject id
  int<lower=1, upper=K> item[N];   //item id
  int<lower=1> fixright;           //item to fix on right
}

parameters {
  vector[J] alpha_raw;         // respondent speech ideology (raw)
  vector[J] beta;              // respondent "outspokenness"
  vector[K] gamma;             // item slants (direction and strength of association with speech ideology)
  vector[K] d4_A;              // additive cutpoint component 1
  vector<lower=0>[K] d4_B;     // additive cutpoint component 2
  vector<lower=0>[K] d4_C;     // additive cutpoint component 3
  real<lower=0.1> sigma_beta;  // outspokenness spread
}

transformed parameters { // hard-standardize alphas to aid model identification
  vector[J] alpha;
  alpha = (alpha_raw - mean(alpha_raw)) ./ sd(alpha_raw); 
}

model {
  vector[3] c;
  //priors
  gamma[fixright] ~ exponential(.1);     // enforce America first right-slanted
  alpha_raw ~ normal(0,1);               // standard normal prior on speech ideology
  beta ~ normal(0,sigma_beta);           // normal prior (mean zero) on statement slants
  
  for (i in 1:N){
    c = [d4_A[item[i]], d4_A[item[i]] + d4_B[item[i]], d4_A[item[i]] + d4_B[item[i]] + d4_C[item[i]]]';  // construct c additively
    say[i] ~ ordered_logistic(                         // logistic link
    gamma[item[i]] * alpha[subj[i]] + beta[subj[i]],   // utility function
    c);                                                // response cutpoints
  }
}

generated quantities {  // transform cutpoints into ideology space for plotting
  vector[K] c4_transformed_1;
  vector[K] c4_transformed_2;
  vector[K] c4_transformed_3;
  c4_transformed_1 = d4_A ./ gamma;
  c4_transformed_2 = (d4_A + d4_B) ./ gamma;
  c4_transformed_3 = (d4_A + d4_B + d4_C) ./ gamma;
}
