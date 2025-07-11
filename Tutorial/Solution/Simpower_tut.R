# Simulation-Based Statistical Inference for HCI – R Tutorial with Visualizations
# -------------------------------------------------------------------
# This tutorial demonstrates how statistical power
# behaves for different statistical tests under different conditions.
# We visualize both the raw data and the p-value distributions.
# -------------------------------------------------------------------
# Key Concepts:
# - Type I error (α): False positive rate when H0 is true.
# - Power (1 - β): True positive rate when there is an effect.
# - Liberal test: Rejects too often (inflated α).
# - Conservative test: Rejects too rarely (low power).
# -------------------------------------------------------------------


set.seed(123)
library(tidyverse)
library(pwr)
library(glue)


n <- 12
theme_set(theme_minimal(base_size = 12))

# -------------------------------------------------------------------------
# Visualize raw simulated dataset from one simulation ---------------------
# -------------------------------------------------------------------------

# Example of a strong effect (for power visualization)
x_example <- rnorm(n, 0, 1)
y_example <- rnorm(n, .5, 1)
df_example <- tibble(value = c(x_example, y_example),
                     group = factor(rep(c("G1", "G2"), each = n)))



  ggplot(df_example, aes(group, value, fill = group)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.7, size = 1.2) +
    labs(title = "Raw data: effect size ~.5", x = NULL, y = "Value") +
    theme(legend.position = "none")






# Fixed effect size and α
d      <- 0.5        # Cohen's d
alpha  <- 0.05
nsim   <- 100

# Sequence of per‐group sample sizes to explore
ns     <- seq(5, 80, by = 5)

# 1) Simulation-based power for each n
sim_powers <- map_dbl(ns, function(n) {
  pvals <- replicate(nsim, {
    x <- rnorm(n, mean = 0, sd = 1)
    y <- rnorm(n, mean = d, sd = 1)
    t.test(x, y, var.equal = TRUE)$p.value
  })
  mean(pvals < alpha)
})

# 2) Analytical power for each n
ana_powers <- map_dbl(ns, function(n) {
  pwr.t.test(
    n          = n,
    d          = d,
    sig.level  = alpha,
    type       = "two.sample",
    alternative = "two.sided"
  )$power
})

# Combine into tidy data frame
power_df <- tibble(
  n       = rep(ns, 2),
  power   = c(sim_powers, ana_powers),
  method  = rep(c("Simulation", "Analytical"), each = length(ns))
)

# Plot
ggplot(power_df, aes(n, power, color = method, shape = method)) +
  geom_line() +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.8, linetype = "dotted") +
  labs(
    title    = glue("Power vs. Sample Size (d = {d}, α = {alpha})"),
    x        = "Sample Size per Group (n)",
    y        = "Power",
    color    = "Method",
    shape    = "Method"
  ) +
  theme(legend.position = "bottom")






## non-parametric data fixed effect size and sample size
### You expect A<B, C is just a control condition
# ------------------------------------------------------------------------------
# 1. Parameters and cut-points for transforming latent normals into 1–5 scale
# ------------------------------------------------------------------------------
n_per_group <- 12        # number of observations per group
shift_AB    <- 0.5       # mean difference between group A and B on the latent scale
shift_C     <- 0.1      # mean difference for group C
cut_points  <- c(-Inf, -0.8, -0.25, 0.25, 0.8, Inf)  # defines thresholds for 5 ordinal levels

# ------------------------------------------------------------------------------
# 2. Generating one set of raw ordinal data for groups A, B, and C
# ------------------------------------------------------------------------------
# 2a. Create latent-normal values for each group
latent_A <- rnorm(n_per_group, mean = 0,      sd = 1)
latent_B <- rnorm(n_per_group, mean = shift_AB, sd = 1)
latent_C <- rnorm(n_per_group, mean = shift_C,  sd = 1)

# 2b. Convert latent values into integer levels 1–5 by cutting at the predefined breaks
value_A <- as.integer(cut(latent_A, breaks = cut_points, labels = FALSE, right = FALSE))
value_B <- as.integer(cut(latent_B, breaks = cut_points, labels = FALSE, right = FALSE))
value_C <- as.integer(cut(latent_C, breaks = cut_points, labels = FALSE, right = FALSE))

# 2c. Build a data.frame with explicit columns
raw_df <- data.frame(
  group = rep(c("A", "B", "C"), each = n_per_group),
  value = factor(
    c(value_A, value_B, value_C),
    levels = 1:5,
    ordered = TRUE
  )
)

# ------------------------------------------------------------------------------
# 3. Inspect the raw data by hand
# ------------------------------------------------------------------------------
# 3a. Frequency counts for each level within each group
freq_table <- table(raw_df$group, raw_df$value)
print("Frequency table of observed levels in each group:")
print(freq_table)

# 3b. Summary statistics (median and interquartile range)
# We coerce the ordered factor back to integer for summary computations
raw_df$value_int <- as.integer(raw_df$value)

summary_stats <- data.frame(
  group = character(0),
  median = numeric(0),
  Q1     = numeric(0),
  Q3     = numeric(0),
  stringsAsFactors = FALSE
)

for (grp in unique(raw_df$group)) {
  this_values <- raw_df$value_int[raw_df$group == grp]
  summary_stats <- rbind(summary_stats, data.frame(
    group = grp,
    median = median(this_values),
    Q1     = quantile(this_values, 0.25),
    Q3     = quantile(this_values, 0.75),
    stringsAsFactors = FALSE
  ))
}

print("Summary statistics by group:")
print(summary_stats)

# ------------------------------------------------------------------------------
# 4. Visualize the distribution of ordinal responses
# ------------------------------------------------------------------------------
# We manually compute proportions for each (group, level) pair and then plot.
prop_df <- as.data.frame(freq_table) %>%
  rename(group = Var1, value = Var2, count = Freq) %>%
  group_by(group) %>%
  mutate(proportion = count / sum(count))

ggplot(data = prop_df, aes(x = group, y = proportion, fill = value)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Distribution of Ordinal Responses by Group",
    x = "Group",
    y = "Proportion of Responses",
    fill = "Response Level"
  ) +
  scale_fill_brewer(palette = "Blues", direction = 1)

# ------------------------------------------------------------------------------
# 5. Simple non-parametric tests on the single dataset
# ------------------------------------------------------------------------------
# 5a. Kruskal–Wallis test (A, B, and C)
kw_test_result <- kruskal.test(value_int ~ group, data = raw_df)
cat("Kruskal–Wallis test (A, B, C): p-value =", round(kw_test_result$p.value, 3), "\n")


# 5b. Wilcoxon rank‐sum (A vs B only)
values_A  <- raw_df$value_int[raw_df$group == "A"]
values_B  <- raw_df$value_int[raw_df$group == "B"]
wilcox_test_result <- wilcox.test(values_A, values_B, exact = FALSE)
cat("Wilcoxon rank‐sum test (A vs B): p-value =", round(wilcox_test_result$p.value, 3), "\n")

# ------------------------------------------------------------------------------
# 6. Simulation study: repeat data generation 1,000 times and estimate power
# ------------------------------------------------------------------------------
n_sims <- 1000

# Preallocate vectors to hold p-values
pvals_wilcox  <- numeric(n_sims)
pvals_kruskal <- numeric(n_sims)

for (sim in seq_len(n_sims)) {
  # Generate latent normals for each group
  latent_A <- rnorm(n_per_group, mean = 0,       sd = 1)
  latent_B <- rnorm(n_per_group, mean = shift_AB, sd = 1)
  latent_C <- rnorm(n_per_group, mean = shift_C,  sd = 1)
  
  # Convert to ordinal
  val_A <- as.integer(cut(latent_A, breaks = cut_points, labels = FALSE, right = FALSE))
  val_B <- as.integer(cut(latent_B, breaks = cut_points, labels = FALSE, right = FALSE))
  val_C <- as.integer(cut(latent_C, breaks = cut_points, labels = FALSE, right = FALSE))
  
  # Wilcoxon for A vs B
  test_w <- wilcox.test(val_A, val_B, exact = FALSE)
  pvals_wilcox[sim] <- test_w$p.value
  
  # Kruskal–Wallis across A, B, C
  all_vals  <- c(val_A, val_B, val_C)
  all_group <- factor(rep(c("A","B","C"), each = n_per_group))
  test_kw <- kruskal.test(all_vals, all_group)
  pvals_kruskal[sim] <- test_kw$p.value
}

# Calculate empirical power at α = 0.05
alpha <- 0.05
power_wilcox  <- mean(pvals_wilcox  < alpha)
power_kruskal <- mean(pvals_kruskal < alpha)

power_results <- data.frame(
  test            = c("Wilcoxon rank‐sum", "Kruskal–Wallis"),
  empirical_power = c(power_wilcox, power_kruskal)
)

print("Estimated statistical power (α = 0.05):")
print(power_results)


# ------------------------------------------------------------------------------
# 7. Simulation‐based power curves for Wilcoxon and Kruskal–Wallis over n --------
# ------------------------------------------------------------------------------

set.seed(456)

# 7a. Define range of sample sizes and number of sims
nsim       <- 1000
ns_vec     <- seq(5, 80, by = 5)
shift_AB   <- 0.5      # same as before
shift_C    <- 0.1
cut_pts    <- cut_points  # reuse your cut_points vector
alpha      <- 0.05

# 7b. Preallocate lists to store power results
power_nonpar <- map_dfr(ns_vec, function(n_per_group) {
  # vectors to collect p‐values
  p_w   <- numeric(nsim)
  p_kw  <- numeric(nsim)
  
  for (i in seq_len(nsim)) {
    # simulate latent normals
    lat_A <- rnorm(n_per_group, mean = 0,        sd = 1)
    lat_B <- rnorm(n_per_group, mean = shift_AB, sd = 1)
    lat_C <- rnorm(n_per_group, mean = shift_C,  sd = 1)
    
    # cut to ordinal
    vA <- as.integer(cut(lat_A, breaks = cut_pts, labels = FALSE, right = FALSE))
    vB <- as.integer(cut(lat_B, breaks = cut_pts, labels = FALSE, right = FALSE))
    vC <- as.integer(cut(lat_C, breaks = cut_pts, labels = FALSE, right = FALSE))
    
    # Wilcoxon A vs B
    p_w[i]  <- wilcox.test(vA, vB, exact = FALSE)$p.value
    
    # Kruskal–Wallis A, B, C
    all_v    <- c(vA, vB, vC)
    grp_fac  <- factor(rep(c("A","B","C"), each = n_per_group))
    p_kw[i]  <- kruskal.test(all_v, grp_fac)$p.value
  }
  
  # compute empirical power
  tibble(
    n        = n_per_group,
    method   = c("Wilcoxon", "Kruskal–Wallis"),
    power    = c(mean(p_w < alpha), mean(p_kw < alpha))
  )
})

# 7c. Plot non‐parametric power curves
ggplot(power_nonpar, aes(x = n, y = power, color = method, shape = method)) +
  geom_line() +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.8, linetype = "dotted") +
  labs(
    title = glue("Non-parametric Power vs. Sample Size (α = {alpha})"),
    x     = "Sample Size per Group (n)",
    y     = "Empirical Power",
    color = "Test",
    shape = "Test"
  ) +
  theme(legend.position = "bottom")





