library(data.table)
library(dHSIC)

printf <- function(...) {
    print(sprintf(...))
}

print_dim <- function(X, name="") {
    printf("%s is (%s)", name, paste(dim(X), collapse=","))
}

listtostr <- function(l, name="") {
    return(sprintf("(%s)", paste(l, collapse=",")))
}

norm <- function(X) {
    return (sum(X^2))
}

generate_data <- function(n=100) {
    noise <- function(d=1) {
        X = matrix(0, n, d)
        for (i in 1:d) {
            X[, i] = runif(n, -1, 1)
        }
        return (X)
    }
    coefs <- function(m) {
        return (t(as.matrix(runif(m, -5, 5))))
    }
    w = runif(6, -5, 5)
    L1 = noise(1)
    L2 = w[1] * L1 + noise(1)
    L3 = w[2] * L1 + w[3] * L2 + noise(1)
    L4 = w[4] * L1 + w[5] * L2 + w[6] * L3 + noise(1)
    X1 = L1 %*% coefs(4) + noise(4)
    X2 = L3 %*% coefs(2) + noise(2)
    X3 = L4 %*% coefs(2) + noise(2)
    X = cbind(X1, X2, X3)
    return (X)
}

gin <- function(X, alpha=0.01) {
    covariance <- function(Y, Z) {
        return (t(Y) %*% Z)
    }

    left_nullspace <- function(E_YZ) {
        obj = svd(t(E_YZ))
        v = obj[["v"]]
        w = as.matrix(v[, ncol(v)])
        w = w / sum(w^2)
        return (w)
    }

    indep_test <- function(E_YllZ, Z, alpha) {
        p_list = c()
        for (i in 1:ncol(Z)) {
            z = Z[, i]
            test = dhsic.test(E_YllZ, z,
                              method = "permutation",
                              kernel = "gaussian", 
                              B = 1000)
            #print(test$p.value)
            p_list = c(p_list, test$p.value)
        }
        k = length(p_list)
        testStat = -2 * sum(log(p_list))
        pval = 1 - pchisq(testStat, 2*k)
        print(pval)
        return (pval > alpha)
    }

    normalize <- function(X) {
        for (i in 1:ncol(X)) {
            X[, i] = X[, i] - mean(X[, i])
        }
        return (X)
    }

    algo1 <- function(X) {
        S = c()
        n = 2
        cols = 1:ncol(X)

        while (TRUE) {
            printf("Testing subsets of dim %i..", n)
            subsets = combn(cols, n)
            for (j in sample(1:ncol(subsets))) {
                q = subsets[, j]
                p = cols[!(cols %in% q)]
                printf("Testing subset %s", listtostr(q))
                Y = X[, q]
                Z = X[, p]
                E_YZ = covariance(Y, Z)
                #print_dim(E_YZ, "E_YZ")
                w = left_nullspace(E_YZ)
                #print_dim(w, "w")
                #print(norm(t(w) %*% E_YZ))
                E_YllZ = Y %*% w
                #print(head(E_YllZ))
                #print_dim(E_YllZ, "E_YllZ")
                indep = indep_test(E_YllZ, Z, alpha=alpha)

                if (indep) {
                    S = c(S, q)
                    q_print = paste(q, collapse=",")
                    printf("(%s) is independent.", q_print)
                }
            }
            n = n + 1
        }
    }

    cols = names(X)
    X = as.matrix(X)
    X = normalize(X)
    algo1(X)
}

X = rnorm(100, 0, 1)
Y = rnorm(100, 0, 1)
Z = 4*X + runif(100, -10, 10)

data <- generate_data(n=1000)
gin(data, alpha=0.05)
