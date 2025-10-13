#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: vina_vs_boltz_analysis.R <boltz_csv> <vina_csv> <output_dir>", call. = FALSE)
}

boltz_csv <- args[1]
vina_csv  <- args[2]
out_dir   <- args[3]

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

repo <- "https://cloud.r-project.org"
required_packages <- c("ggplot2", "dplyr", "readr", "tidyr", "scales", "gridExtra", "broom")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = repo)
  }
}

library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)
library(scales)
library(gridExtra)
library(broom)

boltz <- read_csv(boltz_csv, show_col_types = FALSE) %>%
  mutate(pdbid = toupper(trimws(pdbid)),
         rmsd_locked_global = as.numeric(rmsd_locked_global))

vina <- read_csv(vina_csv, show_col_types = FALSE) %>%
  mutate(pdbid = toupper(trimws(pdbid)),
         rmsd_locked_global = as.numeric(rmsd_locked_global))

df <- inner_join(boltz %>% select(pdbid, rmsd_locked_global_boltz = rmsd_locked_global),
                 vina %>% select(pdbid, rmsd_locked_global_vina = rmsd_locked_global),
                 by = "pdbid") %>%
  drop_na()

if (nrow(df) == 0) stop("Merged data frame is empty.", call. = FALSE)

df <- df %>%
  mutate(diff = rmsd_locked_global_vina - rmsd_locked_global_boltz,
         ratio = rmsd_locked_global_vina / rmsd_locked_global_boltz,
         better_method = if_else(rmsd_locked_global_vina < rmsd_locked_global_boltz, "Vina", "Boltz"))

summary_tbl <- tibble(
  method = c("Boltz", "Vina"),
  mean = c(mean(df$rmsd_locked_global_boltz),
           mean(df$rmsd_locked_global_vina)),
  median = c(median(df$rmsd_locked_global_boltz),
             median(df$rmsd_locked_global_vina)),
  sd = c(sd(df$rmsd_locked_global_boltz),
         sd(df$rmsd_locked_global_vina)),
  mad = c(mad(df$rmsd_locked_global_boltz, constant = 1),
          mad(df$rmsd_locked_global_vina, constant = 1))
)

thresholds <- c(2, 4)
success_tbl <- thresholds %>%
  lapply(function(thr) {
    tibble(
      threshold = thr,
      Boltz = sum(df$rmsd_locked_global_boltz <= thr),
      Vina  = sum(df$rmsd_locked_global_vina  <= thr)
    )
  }) %>%
  bind_rows()

paired_t <- t.test(df$rmsd_locked_global_vina, df$rmsd_locked_global_boltz, paired = TRUE)
wilcox  <- wilcox.test(df$rmsd_locked_global_vina, df$rmsd_locked_global_boltz, paired = TRUE, alternative = "two.sided")
sign_wins_vina <- sum(df$rmsd_locked_global_vina <= df$rmsd_locked_global_boltz)
sign_wins_boltz <- nrow(df) - sign_wins_vina

set.seed(42)
boot_samples <- 10000
boot_means <- replicate(boot_samples, mean(sample(df$diff, replace = TRUE)))
boot_medians <- replicate(boot_samples, median(sample(df$diff, replace = TRUE)))
ci_mean <- quantile(boot_means, c(0.025, 0.975))
ci_median <- quantile(boot_medians, c(0.025, 0.975))

stats_out <- list(
  n = nrow(df),
  summary = summary_tbl,
  success = success_tbl,
  paired_t = broom::tidy(paired_t),
  wilcoxon = broom::tidy(wilcox),
  sign_test = list(wins_vina = sign_wins_vina, wins_boltz = sign_wins_boltz),
  diff_mean = mean(df$diff),
  diff_median = median(df$diff),
  diff_sd = sd(df$diff),
  diff_ci_mean = ci_mean,
  diff_ci_median = ci_median
)

saveRDS(stats_out, file = file.path(out_dir, "stats.rds"))
write_csv(summary_tbl, file.path(out_dir, "summary_statistics.csv"))
write_csv(success_tbl, file.path(out_dir, "success_counts.csv"))
write_csv(df, file.path(out_dir, "merged_data.csv"))

pal <- c("#4C72B0", "#DD8452")

p_scatter <- ggplot(df, aes(x = rmsd_locked_global_boltz, y = rmsd_locked_global_vina)) +
  geom_point(alpha = 0.7, color = "#4C72B0") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#888888") +
  labs(title = "Locked RMSD: Boltz vs Vina",
       x = "Boltz RMSD (Å)",
       y = "Vina RMSD (Å)") +
  coord_equal() +
  theme_minimal(base_size = 12)

p_hist <- ggplot(df, aes(x = diff)) +
  geom_histogram(bins = 20, fill = "#DD8452", color = "white", alpha = 0.8) +
  geom_vline(xintercept = mean(df$diff), color = "#4C72B0", linetype = "dashed") +
  labs(title = "Distribution of Paired RMSD Differences",
       x = "Vina - Boltz (Å)",
       y = "Count") +
  theme_minimal(base_size = 12)

df_long <- df %>%
  select(pdbid, Boltz = rmsd_locked_global_boltz, Vina = rmsd_locked_global_vina) %>%
  tidyr::pivot_longer(cols = c(Boltz, Vina), names_to = "method", values_to = "rmsd")

p_box <- ggplot(df_long, aes(x = method, y = rmsd, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = pal) +
  labs(title = "Locked RMSD Distribution",
       x = "",
       y = "RMSD (Å)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

succ_long <- success_tbl %>%
  tidyr::pivot_longer(cols = c(Boltz, Vina), names_to = "method", values_to = "successes")

p_bar <- ggplot(succ_long, aes(x = factor(threshold), y = successes, fill = method)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = pal) +
  labs(title = "Success Counts by RMSD Threshold",
       x = "Threshold (Å)",
       y = "Number of Targets") +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "scatter.png"), p_scatter, width = 6, height = 6, dpi = 300)
ggsave(file.path(out_dir, "diff_histogram.png"), p_hist, width = 6, height = 4, dpi = 300)
ggsave(file.path(out_dir, "boxplot.png"), p_box, width = 4.5, height = 4.5, dpi = 300)
ggsave(file.path(out_dir, "success_bar.png"), p_bar, width = 5.5, height = 4, dpi = 300)

gridExtra::grid.arrange(p_scatter, p_hist, p_box, p_bar, ncol = 2)
ggsave(file.path(out_dir, "summary_panel.png"), width = 9, height = 7, dpi = 300)
