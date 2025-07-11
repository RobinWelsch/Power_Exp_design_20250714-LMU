# install.packages(c("BayesFactor","tidyverse","gganimate","transformr"))
library(BayesFactor)
library(tidyverse)
library(gganimate)

set.seed(42)
d         <- 0.5               # true Cohen's d
sigma     <- 1                 # within-group SD
n_sims    <- 1000              # sims per cell
priors    <- c(0.3, 0.707, 1)   # Cauchy scale parameters
Ns        <- seq(10, 100, by=10)
pd_thresh <- 0.95               # posterior probability threshold

results <- expand_grid(prior = priors, N = Ns) %>%
  group_by(prior, N) %>%
  summarize(
    power = mean(
      replicate(n_sims, {
        # simulate two independent samples under d
        x <- rnorm(N, mean = 0,           sd = sigma)
        y <- rnorm(N, mean = d * sigma,   sd = sigma)
        # fit Bayesian t-test and draw posterior samples of the effect
        bf        <- ttestBF(x = y, y = x, rscale = prior, paired = FALSE)
        post_draw <- posterior(bf, iterations = 500)  # returns matrix w/ delta samples
        delta_s   <- post_draw[ , "delta"]             # extract the 'delta' column
        mean(delta_s > 0) > pd_thresh
      })
    ),
    .groups = "drop"
  )

# static ggplot
p <- ggplot(results, aes(x = N, y = power, color = factor(prior))) +
  geom_line(size = 1) +
  scale_y_continuous(limits = c(0,1)) +
  labs(
    title    = "Bayesian Power (P(δ>0|data)>0.95) vs. Sample Size",
    subtitle = "True effect d = 0.5, σ = 1",
    x        = "Sample size per group (N)",
    y        = "Estimated Bayesian power",
    color    = "Cauchy prior\nscale"
  ) +
  
  theme(
    plot.background       = element_rect(fill = "black", colour = NA),
    panel.background      = element_rect(fill = "black", colour = NA),
    panel.grid.major      = element_line(colour = "gray30"),
    panel.grid.minor      = element_line(colour = "gray20"),
    axis.text             = element_text(colour = "white"),
    axis.title            = element_text(colour = "white"),
    plot.title            = element_text(colour = "white", size = 16, face = "bold"),
    plot.subtitle         = element_text(colour = "white", size = 12),
    legend.background     = element_rect(fill = "black", colour = NA),
    legend.key            = element_rect(fill = "black", colour = NA),
    legend.text           = element_text(colour = "white"),
    legend.title          = element_text(colour = "white")
  )
p


# Load required libraries
library(ggplot2)
library(gganimate)

# Function to compute t-test posterior parameters for Normal prior
posterior_t <- function(mu0, tau0, n, ybar, sigma) {
  # Prior: Normal(mu0, tau0^2)
  # Likelihood: Normal(ybar, (sigma/sqrt(n))^2)
  # Posterior: Normal(mu_n, tau_n^2)
  tau_n_sq <- 1 / (1/tau0^2 + n/sigma^2)
  mu_n <- tau_n_sq * (mu0/tau0^2 + n*ybar/sigma^2)
  return(list(mu = mu_n, tau = sqrt(tau_n_sq)))
}

# Simulate data
set.seed(123)
true_mu <- 1.0      # true effect
sigma <- 1.0        # known sd
max_n <- 50         # maximum sample size

# Define priors: weak and strong
priors <- data.frame(
  prior = c("Weak", "Strong"),
  mu0 = c(0, 0),
  tau0 = c(5, 0.2)
)

# Simulate sample means ybar for n = 1 to max_n
sim <- data.frame()
for (n in 1:max_n) {
  y <- rnorm(n, mean = true_mu, sd = sigma)
  ybar <- mean(y)
  for (i in 1:nrow(priors)) {
    pr <- priors[i,]
    post <- posterior_t(pr$mu0, pr$tau0, n, ybar, sigma)
    sim <- rbind(sim, data.frame(
      n = n,
      prior = pr$prior,
      mu0 = pr$mu0,
      tau0 = pr$tau0,
      ybar = ybar,
      mu_n = post$mu,
      tau_n = post$tau
    ))
  }
}

# Construct a grid of distributions
plot_data <- do.call(rbind, lapply(1:nrow(sim), function(i) {
  row <- sim[i,]
  x <- seq(-3, 5, length.out = 200)
  prior_density <- dnorm(x, mean = row$mu0, sd = row$tau0)
  post_density <- dnorm(x, mean = row$mu_n, sd = row$tau_n)
  data.frame(
    n = row$n,
    prior = row$prior,
    x = x,
    density = c(prior_density, post_density),
    dist = rep(c("Prior", "Posterior"), each = length(x))
  )
}))

# Plot and animate
p <- ggplot(plot_data, aes(x = x, y = density, color = dist)) +
  geom_line(size = 1) +
  facet_wrap(~ prior) +
  labs(title = 'n = {closest_state}', x = 'mu', y = 'Density') +
  theme(
    plot.background       = element_rect(fill = "black", colour = NA),
    panel.background      = element_rect(fill = "black", colour = NA),
    panel.grid.major      = element_line(colour = "gray30"),
    panel.grid.minor      = element_line(colour = "gray20"),
    axis.text             = element_text(colour = "white"),
    axis.title            = element_text(colour = "white"),
    plot.title            = element_text(colour = "white", size = 16, face = "bold"),
    plot.subtitle         = element_text(colour = "white", size = 12),
    legend.background     = element_rect(fill = "black", colour = NA),
    legend.key            = element_rect(fill = "black", colour = NA),
    legend.text           = element_text(colour = "white"),
    legend.title          = element_text(colour = "white")
  )+
  transition_states(n, transition_length = 2, state_length = 1, wrap = FALSE) +
  ease_aes('cubic-in-out')

# Animate and save
anim <- animate(p, nframes = max_n * 3, fps = 10, width = 600, height = 400)
anim_save('ttest_prior_posterior.gif', animation = anim)


# Load required libraries
library(ggplot2)
library(gganimate)

# Parameters
t_max <- 200           # maximum sample size per scenario
true_delta <- .2     # small effect for t-test and correlation
set.seed(456)

# Scenarios
scenarios <- c("Effect", "Null")
df_list <- list()

tests <- c("One-sample t-test", "Correlation test")

for (scenario in scenarios) {
  delta <- if (scenario == "Effect") true_delta else 0
  rho   <- if (scenario == "Effect") true_delta else 0
  
  # Preallocate
  p_t <- rep(NA, t_max)
  p_c <- rep(NA, t_max)
  
  y <- numeric(t_max)
  x <- numeric(t_max)
  z <- numeric(t_max)
  
  for (i in seq_len(t_max)) {
    # One-sample t-test data
    y[i] <- rnorm(1, mean = delta, sd = 1)
    if (i > 1) p_t[i] <- t.test(y[1:i], mu = 0)$p.value
    
    # Correlation test data
    
    x[i] <- rnorm(1)
    z[i] <- rho * x[i] + sqrt(1 - (rho)^2) * rnorm(1)
    if (i > 2) p_c[i] <- cor.test(x[1:i], z[1:i])$p.value
  }
  
  df_list[[scenario]] <- data.frame(
    n        = rep(1:t_max, 2),
    p_value  = c(p_t, p_c),
    test     = rep(tests, each = t_max),
    scenario = scenario
  )
}

# Combine data
data_plot <- do.call(rbind, df_list)

# Identify Type I and Type II errors
data_plot$type_error <- with(data_plot, ifelse(
  scenario == "Null" & p_value < 0.05, "Type I",
  ifelse(scenario == "Effect" & p_value > 0.05, "Type II", NA)
))

# Plot setup
g <- ggplot(data_plot, aes(x = n, y = p_value, color = scenario)) +
  # Static horizontal threshold line at alpha = 0.05
  geom_hline(yintercept = 0.05,
             color = "red", size = 1, linetype = "dashed",
             inherit.aes = FALSE) +
  # Draw p-value trajectories
  geom_line(aes(linetype = scenario), size = 1) +
  # Persistent error points
  geom_point(aes(shape = type_error,fill="white"), size = 5, na.rm = TRUE) +
  # Scales
  scale_color_manual(values = c(Effect = "steelblue", Null = "darkorange")) +
  scale_linetype_manual(values = c(Effect = "dashed", Null = "solid")) +
  scale_shape_manual(values = c("Type I" = 4, "Type II" = 1), na.translate = FALSE) +
  scale_fill_manual(values = "white") +
  
  # Facets
  facet_wrap(~ test, ncol = 1) +
  # Labels and theme
  labs(
    title    = 'Sample size: {frame_along}',
    x        = 'Sample size (n)',
    y        = 'p-value',
    color    = 'Scenario',
    linetype = 'Scenario',
    shape    = 'Error'
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background  = element_rect(fill = "black", colour = NA),
    panel.background = element_rect(fill = "black", colour = NA),
    panel.grid.major = element_line(colour = "gray30"),
    panel.grid.minor = element_line(colour = "gray20"),
    axis.text        = element_text(colour = "white"),
    axis.title       = element_text(colour = "white"),
    strip.text       = element_text(colour = "white"),
    legend.background= element_rect(fill = "black", colour = NA),
    legend.text      = element_text(colour = "white"),
    legend.title     = element_text(colour = "white")
  ) +
  ylim(0, 1) +
  # Animation with persistent past for points
  # Static threshold line and cumulative animation without shadow_mark
  transition_manual(frames = n, cumulative = TRUE)

# Animate and save the GIF
# Note: Ensure the number of frames matches t_max
gif <- animate(g, nframes = t_max, fps = 8, width = 800, height = 500)
anim_save("pvalue_type_errors_persistent.gif", gif)





# --- load packages ---
library(ggplot2)
library(tidyr)
library(scales)

# --- parameters ---
alpha <- 0.05
powers <- c("80% power" = 0.80,
            "20% power" = 0.20,
            "10% power" = 0.10)

# sequence of pre-study odds R from 0 to 1
R <- seq(0, 1, length.out = 1001)

# build a data-frame with one column per power
df <- data.frame(R = R)
for (nm in names(powers)) {
  p <- powers[[nm]]
  df[[nm]] <- p * R / (p * R + alpha)
}

# reshape into long format for ggplot
df_long <- pivot_longer(df,
                        cols      = -R,
                        names_to  = "Power",
                        values_to = "PostStudyProb")

# plot
p <- ggplot(df_long, aes(x = R, y = PostStudyProb, color = Power)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = c("80% power" = "#1f78b4",
                                "20% power" = "#33a02c",
                                "10% power" = "#ff7f00")) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  labs(
    x     = "Pre-study odds R",
    y     = "Post-study probability (%)",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position   = c(0.75, 0.25),
    legend.background = element_rect(fill = alpha("white", 0.7), colour = NA)
  )

# display
print(p)

# optionally save to file
# ggsave("ppv_curves.png", p, width = 6, height = 4, dpi = 300)


