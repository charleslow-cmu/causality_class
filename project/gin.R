if (!("pacman" %in% installed.packages()[, "Package"])) {
    install.packages("pacman", dependencies=TRUE, quietly=TRUE)
}
pacman::p_load(data.table, dHSIC, dplyr, iterators, parallel, 
               foreach, doParallel)

printf <- function(..., debug=FALSE) {
    tryCatch({
        if (debug) print(sprintf(...))
    }, error=function(cond) {
        print(sprintf(...))
        stop("ERROR!")
    })
}

print_dim <- function(X, name="") {
    printf("%s is (%s)", name, paste(dim(X), collapse=","), debug=TRUE)
}

listtostr <- function(l, name="") {
    return(sprintf("(%s)", paste(l, collapse=",")))
}

dicttostr <- function(d) {
    m = "{ "
    for (i in names(d)) {
        m = paste0(m, sprintf("%s:%s ", i, listtostr(d[[i]])))
    }
    m = paste0(m, "}")
    return(m)
}

norm <- function(X) {
    return (sum(X^2))
}

savelist <- function(S) {
    dt = strftime(Sys.time(), "%Y-%m-%d-%H-%M")
    fn <- sprintf('logs/S-%s.txt', dt)
    for (i in 1:length(S)) {
        writeLines(sprintf("%i: %s", i, listtostr(S[[i]])), fn)
    }
}

generate_data <- function(n=100, level=1) {
    noise <- function(d=1) {
        X = matrix(0, n, d)
        for (i in 1:d) {
            X[, i] = runif(n, -1, 1)
        }
        return (X)
    }
    coefs <- function(m) {
        sign = sample(c(-1, 1), m, replace=TRUE)
        magnitude = runif(m, 0.5, 2)
        return (t(as.matrix(sign * magnitude)))
    }
    w = c(coefs(6))
    L1 = noise(1)
    L2 = w[1] * L1 + noise(1)
    L3 = w[2] * L1 + w[3] * L2 + noise(1)
    L4 = w[4] * L1 + w[5] * L2 + w[6] * L3 + noise(1)

    if (level == 3) {
        X1 = L1 %*% coefs(4) + L2 %*% coefs(4) + noise(4)
        X2 = L3 %*% coefs(2) + noise(2)
        X3 = L4 %*% coefs(2) + noise(2)
        X = cbind(X1, X2, X3)
    }

    else if (level == 1) {
        X1 = L1 %*% coefs(2) + noise(2)
        X2 = L2 %*% coefs(2) + noise(2)
        X = cbind(X1, X2)
    }
    print_dim(X, "X")
    return (X)
}


gin <- function(X, alpha=0.01, debug=FALSE) {
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
            p_list = c(p_list, test$p.value)
        }
        k = length(p_list)
        testStat = -2 * sum(log(p_list))
        pval = 1 - pchisq(testStat, 2*k)
        return (pval > alpha)
    }

    normalize <- function(X) {
        for (i in 1:ncol(X)) {
            X[, i] = X[, i] - mean(X[, i])
            #X[, i] = X[, i] / sd(X[, i])
        }
        return (X)
    }

    algo1 <- function(X, colnames) {
        S = list()
        si = 1
        s = c()
        n = 2
        cols = 1:ncol(X)

        while (TRUE) {
            subsets = combn(cols, n)
            printf("Testing subsets of dim %i..: %i sets", 
                   n, length(subsets), debug=debug)
            jlist = sample(1:ncol(subsets))

            r = foreach (j = jlist, .combine=rbind) %dopar% {
                q = subsets[, j]
                Log(sprintf("Testing %s...", listtostr(q)))
                p = cols[!(cols %in% q)]
                Y = X[, q]
                Z = X[, p]
                E_YZ = covariance(Y, Z)
                w = left_nullspace(E_YZ)
                E_YllZ = Y %*% w
                indep = indep_test(E_YllZ, Z, alpha=alpha)
                if (indep) {
                    Log(sprintf("%s is independent", listtostr(q)))
                }
                list(q, indep)
            }

            # Update S 
            for (i in 1:nrow(r)) {
                q = r[i,1][[1]]
                indep = r[i,2][[1]]
                if (indep) {
                    S[[si]] = q
                    si = si+1
                    s = union(s, q)
                }
            }
            n = n + 1
            S = merge_sets(S, debug=debug)
            cols = setdiff(cols, s)
            if (length(cols) <= n) break
        }
        return (S)
    }

    colnames = names(X)
    X = as.matrix(X)
    X = normalize(X)
    S = algo1(X, colnames)
    return (S)
}


# S: list of vectors
merge_sets <- function(S, debug=FALSE) {
    J = list(); j = 1
    d = list()
    for (s in S) {
        s = as.character(s)
        printf("Starting %s...", listtostr(s), debug=debug)
        overlap = intersect(s, names(d))
        if (length(overlap) > 0) {
            keys = sapply(as.vector(unlist(overlap)), function(x) d[[x]])
            keys = unique(keys)
            printf("Overlap of %s, overlapping keys are: %s", 
                   listtostr(s), listtostr(keys), debug=debug)
            key = keys[1]

            # Add s to existing key
            tryCatch({
                J[[key]] = union(J[[key]], s)
                for (i in s) {
                    d[[i]] = key
                }
                printf("Merging %s to %s", 
                       listtostr(s), key, debug=debug)
                printf("J is now %s", dicttostr(J), debug=debug)
            }, error=function(cond) {
                printf("J is now %s", dicttostr(J), debug=debug)
                printf("d is now %s", dicttostr(d), debug=debug)
                traceback()
                stop("ERROR!")
            })

            # Add all overlapping sets to existing key
            if (length(keys) > 1) {
                for (oi in 2:length(keys)) {
                    okey = keys[oi]
                    printf("Merging %s to %s", 
                           listtostr(J[[okey]]), key, debug=debug)
                    J[[key]] = union(J[[key]], J[[okey]])
                    for (ovalue in J[[okey]]) {
                        d[[ovalue]] = key
                    }
                    J[[okey]] = NULL
                    printf("J is now %s", dicttostr(J), debug=debug)
                }
            }
        }

        else {
            J[[as.character(j)]] = s
            for (i in s) {
                d[[i]] = as.character(j)
            }
            j = j+1
        }
    }
    return(unname(J))
}

log.socket <- make.socket(port=4000)
Log <- function(text, ...) {
  msg <- sprintf(paste0(as.character(Sys.time()), ": ", text, "\n"), ...)
  cat(msg)
  write.socket(log.socket, msg)
}

numCores <- detectCores()
registerDoParallel(numCores)
#data = generate_data(n=50, level=3)
data = fread("data/cleaned.csv") 
data = data[revenue > 0]
data = data %>% select(-all_of(c("title", "gTV Movie"))) 
data = sample_n(data, 500)
S = gin(data, alpha=0.05, debug=TRUE)
print(S)
savelist(S)
