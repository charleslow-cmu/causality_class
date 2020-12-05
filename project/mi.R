
# (0,0), (0,1), (1,0), (1,1)
get_cpd <- function(size=2, probs) {
  cpds = c()
  positives = sample(probs, 2, replace=TRUE)
  for (pos in positives) {
    cpds = c(cpds, c(pos, 1-pos))
  }
  return (cpds)
}

probs = seq(0.1, 0.9, length.out = 9)
L1_X1 = get_cpd(2, probs)