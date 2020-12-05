pacman::p_load(data.table, poLCA, dplyr, ggplot2, dendextend)

make_formula <- function(cols) {
  txt = sprintf("cbind(%s) ~ 1", paste(cols, collapse=","))
  return (as.formula(txt))
}

get_dataframe <- function(m) {
  df = Reduce(rbind, m$probs)
  df = data.table(df)
  R = nrow(m$probs[[1]])
  J = length(names(m$probs))
  df[, var := rep(names(m$probs), each=R)]
  df[, class := paste0("c", rep(1:R, J))]
  return (df)
}

get_class_probs <- function(df) {
  df[, prob := `Pr(2)`]
  df = df %>% select(class, var, prob)
  df[, prob := prob / sum(prob), by=.(var)]
  df = df %>% dcast(var ~ class, value.var="prob")
  namelist = df$var
  df = df %>%
        select(-var) %>%
        as.matrix()
  row.names(df) = namelist
  return (df)
}

plot_dendrogram <- function(cols) {
  m <- poLCA(make_formula(cols), data=data, nclass=4, maxiter=5000)
  df = get_dataframe(m)
  class_probs = get_class_probs(df)
  d <- dist(class_probs, method="euclidean")
  hc <- hclust(d, method="complete")
  plot(hc, cex=1.0)
}

data = fread("data/cleaned.csv")
data = data[revenue > 0]
data = data %>% select(-all_of(c("title", "revenue")))
data = data + 1
cast_cols = grep("^a", names(data), value=TRUE)
genre_cols = c("gAction", "gComedy", "gDrama", "gFamily", "gCrime")

m <- poLCA(make_formula(cast_cols), data=data, nclass=4, maxiter=5000)
png(filename="plots/lca.png", width=1200, height=800, pointsize = 20)
plot(m)
dev.off()

df = get_dataframe(m)

