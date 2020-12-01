library(data.table)
library(ggplot2)
suppressMessages(library(dplyr))

data = fread("data/cleaned.csv")
data = data[revenue > 0]  # Remove 0 revenue movies
test_data = rlang::duplicate(data)
genre_cols = grep("^g", names(data), value=TRUE)
genre_cols = genre_cols[genre_cols != "gTV Movie"] # No entries
actor_cols = grep("^a", names(data), value=TRUE)

plot_residuals <- function(data, ycol, xcols, type="scatter") {
    temp = data %>% select(all_of(ycol), all_of(xcols))
    eqn = paste0(ycol, " ~ .")
    m = lm(eqn, temp)
    test_data$predicted = predict(m, test_data)
    test_data$residuals = test_data$revenue - test_data$predicted

    if (type == "scatter") {
        p = ggplot(test_data, aes(x=predicted, y=residuals)) +
            geom_point(alpha=0.3)
    }
    if (type == "hist") {
        p = ggplot(test_data, aes(x=residuals)) + 
            geom_density()
    }
    return (p)
}


p = plot_residuals(data, "revenue", genre_cols, type="scatter") +
    labs(title="Residuals vs Fitted",
    subtitle="From Regressing Revenue on Genre")
ggsave(plot=p, "plots/genre.png", width=8, height=5)

p = plot_residuals(data, "revenue", genre_cols, type="hist") +
    labs(title="Histogram of Residuals",
    subtitle="From Regressing Revenue on Genre")
ggsave(plot=p, "plots/genre_hist.png", width=8, height=5)


p = plot_residuals(data, "revenue", actor_cols) +
    labs(title="Residuals vs Fitted",
    subtitle="From Regressing Revenue on Cast")
ggsave(plot=p, "plots/cast.png", width=8, height=5)

p = plot_residuals(data, "revenue", actor_cols, type="hist") +
    labs(title="Histogram of Residuals",
    subtitle="From Regressing Revenue on Cast")
ggsave(plot=p, "plots/cast_hist.png", width=8, height=5)
