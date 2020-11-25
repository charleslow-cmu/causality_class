library(ggplot2)
library(gridExtra)
setwd("~/Desktop/causality_class/assignment3/")

n = 1000
a = 0.5
X = rnorm(n, 0, 1)
E = rnorm(n, 0, 1)
Y = a * X + E
df = data.frame(X = X, Y = Y, E = E)

# a1
p1 = ggplot(df, aes(x=X, y=Y)) + geom_point(alpha=0.7) + ggtitle("Scatterplot X on Y")
p2 = ggplot(df, aes(x=X, y=E)) + geom_point(alpha=0.7) + ggtitle("Scatterplot X on E")
p = grid.arrange(p1, p2, nrow=2)
ggsave(p, filename="a1.png", width=6, height=7)

# a2
m1 = lm(data=df, "Y ~ X")
df$residual1 = m1$residuals
p3 = ggplot(df, aes(x=X, y=residual1)) + geom_point(alpha=0.7) + 
  ggtitle("Scatterplot X on Residual") +
  labs(y="Residual")
p3
ggsave(p3, filename="a2.png", width=6, height=4)

# a3
m2 = lm(data=df, "X ~ Y")
df$residual2 = m2$residuals
p4 = ggplot(df, aes(x=Y, y=residual2)) + geom_point(alpha=0.7) + 
  ggtitle("Scatterplot Y on Residual") +
  labs(y="Residual")
p4
ggsave(p4, filename="a3.png", width=6, height=4)


########################################################################

n = 1000
a = 0.5
X = rnorm(n, 0, 1)
E = runif(n, -1, 1)
Y = a * X + E
df = data.frame(X = X, Y = Y, E = E)

# b1
p1 = ggplot(df, aes(x=X, y=Y)) + geom_point(alpha=0.7) + ggtitle("Scatterplot X on Y")
p2 = ggplot(df, aes(x=X, y=E)) + geom_point(alpha=0.7) + ggtitle("Scatterplot X on E")
p = grid.arrange(p1, p2, nrow=2)
ggsave(p, filename="b1.png", width=6, height=7)

# b2
m1 = lm(data=df, "Y ~ X")
df$residual1 = m1$residuals
p3 = ggplot(df, aes(x=X, y=residual1)) + geom_point(alpha=0.7) + 
  ggtitle("Scatterplot X on Residual") +
  labs(y="Residual")
p3
ggsave(p3, filename="b2.png", width=6, height=4)

# b3
m2 = lm(data=df, "X ~ Y")
df$residual2 = m2$residuals
p4 = ggplot(df, aes(x=Y, y=residual2)) + geom_point(alpha=0.7) + 
  ggtitle("Scatterplot Y on Residual") +
  labs(y="Residual")
p4
ggsave(p4, filename="b3.png", width=6, height=4)

setwd("../assignment2")
library(data.table)
df = fread("heart.csv")
names(df)
df <- df[, c("age", "sex", "chol", "trestbps")]
fwrite(df, "heart_subset.csv")
