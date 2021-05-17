from GaussianGraph import GaussianGraph



def scenario0a():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", ["L1"])

    g.add_variable("X1", "L1")
    g.add_variable("X2", "L1")
    g.add_variable("X3", "L2")
    g.add_variable("X4", "L2")
    return g


def scenario0b():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", ["L1"])

    g.add_variable("X1", ["L1", "L2"])
    g.add_variable("X2", ["L1", "L2"])
    g.add_variable("X3", ["L1", "L2"])
    g.add_variable("X4", ["L1", "L2"])
    g.add_variable("X5", ["L1", "L2"])
    g.add_variable("X6", ["L1", "L2"])
    return g

def scenario0c():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", ["L1"])

    g.add_variable("X1", "L2")
    g.add_variable("X2", "L2")
    g.add_variable("X3", "L2")
    g.add_variable("X4", ["L1", "L2"])
    g.add_variable("X5", ["L1", "L2"])
    g.add_variable("X6", ["L1", "L2"])
    return g

def scenario1():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", ["L1"])
    g.add_variable("L3", ["L1", "L2"])
    g.add_variable("L4", ["L1", "L2", "L3"])
    g.add_variable("L5", ["L1", "L2", "L3", "L4"])
    g.add_variable("L6", ["L1", "L2", "L3", "L4", "L5"])

    g.add_variable("X1", "L1")
    g.add_variable("X2", "L1")
    g.add_variable("X3", ["L2", "L3"])
    g.add_variable("X4", ["L2", "L3"])
    g.add_variable("X5", ["L2", "L3"])
    g.add_variable("X6", ["L4", "L5"])
    g.add_variable("X7", ["L4", "L5"])
    g.add_variable("X8", ["L4", "L5"])
    g.add_variable("X9", "L6")
    g.add_variable("X10", "L6")
    return g

def scenario2():
    g = GaussianGraph()
    g.add_variable("L3", None)
    g.add_variable("L1", "L3")
    g.add_variable("L2", ["L1", "L3"])

    g.add_variable("X1", "L1")
    g.add_variable("X2", "L1")
    g.add_variable("X3", "L2")
    g.add_variable("X4", "L2")
    g.add_variable("X5", ["L1", "L2"])
    g.add_variable("X6", "L3")
    g.add_variable("X7", "L3")
    g.add_variable("X8", ["L2", "L3"])
    return g



def scenario3a():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L1")
    g.add_variable("L6", "L2")
    g.add_variable("L7", ["L2", "L3"])
    g.add_variable("L8", ["L2", "L3", "L7"])
    g.add_variable("L9", ["L2", "L3"])
    g.add_variable("L10", ["L3", "L9"])
    g.add_variable("L11", "L4")
    g.add_variable("L12", ["L4", "L5"])
    g.add_variable("L13", ["L4", "L5", "L12"])
    g.add_variable("L14", ["L4", "L5"])
    g.add_variable("L15", ["L5", "L14"])
    g.add_variable("X1", "L6")
    g.add_variable("X2", "L6")
    g.add_variable("X3", "L7")
    g.add_variable("X4", ["L7", "L8"])
    g.add_variable("X5", ["L7", "L8"])
    g.add_variable("X6", "L9")
    g.add_variable("X7", "L9")
    g.add_variable("X8", ["L9", "L10"])
    g.add_variable("X9", "L10")
    g.add_variable("X10", "L11")
    g.add_variable("X11", "L11")
    g.add_variable("X12", "L12")
    g.add_variable("X13", ["L12", "L13"])
    g.add_variable("X14", ["L12", "L13"])
    g.add_variable("X15", "L14")
    g.add_variable("X16", "L14")
    g.add_variable("X17", ["L14", "L15"])
    g.add_variable("X18", "L15")
    return g



def scenario3b():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L1")
    g.add_variable("L6", ["L2", "L3"])
    g.add_variable("L7", ["L2", "L3", "L6"])
    g.add_variable("L8", ["L2", "L3", "L7"])
    g.add_variable("L9", ["L2", "L3"])
    g.add_variable("L10", ["L3", "L9"])
    g.add_variable("L11", ["L4", "L5"])
    g.add_variable("L12", ["L4", "L5", "L11"])
    g.add_variable("L13", ["L4", "L5", "L12"])
    g.add_variable("L14", ["L4", "L5"])
    g.add_variable("L15", ["L5", "L14"])
    g.add_variable("X1", "L6")
    g.add_variable("X2", "L6")
    g.add_variable("X3", "L7")
    g.add_variable("X4", ["L7", "L8"])
    g.add_variable("X5", ["L7", "L8"])
    g.add_variable("X6", "L9")
    g.add_variable("X7", "L9")
    g.add_variable("X8", ["L9", "L10"])
    g.add_variable("X9", "L10")
    g.add_variable("X10", "L11")
    g.add_variable("X11", "L11")
    g.add_variable("X12", "L12")
    g.add_variable("X13", ["L12", "L13"])
    g.add_variable("X14", ["L12", "L13"])
    g.add_variable("X15", "L14")
    g.add_variable("X16", "L14")
    g.add_variable("X17", ["L14", "L15"])
    g.add_variable("X18", "L15")
    return g



def scenario3c():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L1")
    g.add_variable("L6", "L2")
    g.add_variable("L7", ["L2", "L3", "L6"])
    g.add_variable("L8", ["L2", "L3", "L7"])
    g.add_variable("L9", ["L2", "L3"])
    g.add_variable("L10", ["L3", "L9"])
    g.add_variable("L11", "L4")
    g.add_variable("L12", ["L4", "L5", "L11"])
    g.add_variable("L13", ["L4", "L5", "L12"])
    g.add_variable("L14", ["L4", "L5"])
    g.add_variable("L15", ["L5", "L14"])

    g.add_variable("X1", ["L6", "L7", "L8"])
    g.add_variable("X2", ["L6", "L7", "L8"])
    g.add_variable("X3", ["L6", "L7", "L8"])
    g.add_variable("X4", ["L6", "L7", "L8"])
    g.add_variable("X5", ["L6", "L7", "L8"])
    g.add_variable("X6", "L9")
    g.add_variable("X7", "L9")
    g.add_variable("X8", ["L9", "L10"])
    g.add_variable("X9", "L10")
    g.add_variable("X10", ["L11", "L12", "L13"])
    g.add_variable("X11", ["L11", "L12", "L13"])
    g.add_variable("X12", ["L11", "L12", "L13"])
    g.add_variable("X13", ["L11", "L12", "L13"])
    g.add_variable("X14", ["L11", "L12", "L13"])
    g.add_variable("X15", "L14")
    g.add_variable("X16", "L14")
    g.add_variable("X17", ["L14", "L15"])
    g.add_variable("X18", "L15")
    return g


def scenario4():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L1")
    g.add_variable("L6", "L2")
    g.add_variable("L7", ["L2", "L3"])
    g.add_variable("L8", ["L2", "L3", "L7"])
    g.add_variable("L9", ["L2", "L3"])
    g.add_variable("L10", ["L3", "L9"])

    g.add_variable("X1", "L6")
    g.add_variable("X2", "L6")
    g.add_variable("X3", "L7")
    g.add_variable("X4", ["L7", "L8"])
    g.add_variable("X5", ["L7", "L8"])
    g.add_variable("X6", "L9")
    g.add_variable("X7", "L9")
    g.add_variable("X8", ["L9", "L10"])
    g.add_variable("X9", "L10")
    g.add_variable("X10", "L2")
    g.add_variable("X11", "L3")
    g.add_variable("X12", ["L4", "L5"])
    g.add_variable("X13", ["L4", "L5"])
    g.add_variable("X14", ["L4", "L5"])
    g.add_variable("X15", "L5")
    return g


def scenario5a():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", None)
    g.add_variable("L3", "L1")
    g.add_variable("L4", ["L1", "L2"])
    g.add_variable("L5", "L2")
    g.add_variable("L6", "L1")
    g.add_variable("L7", ["L1", "L2"])
    g.add_variable("L8", ["L1", "L2"])

    g.add_variable("X1", ["L3", "L4", "L5"])
    g.add_variable("X2", ["L3", "L4", "L5"])
    g.add_variable("X3", ["L3", "L4", "L5"])
    g.add_variable("X4", ["L3", "L4", "L5"])
    g.add_variable("X5", "L1")
    g.add_variable("X6", "L2")
    g.add_variable("X7", "L6")
    g.add_variable("X8", "L6")
    g.add_variable("X9", ["L7", "L8"])
    g.add_variable("X10", ["L7", "L8"])
    g.add_variable("X11", ["L7", "L8"])
    return g



def scenario5b():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", None)
    g.add_variable("L3", ["L1", "L2"])
    g.add_variable("L4", ["L1", "L2"])
    g.add_variable("L5", ["L1", "L2"])
    g.add_variable("L6", ["L1", "L2"])
    g.add_variable("L7", ["L1", "L2"])
    g.add_variable("L8", ["L1", "L2"])

    g.add_variable("X1", "L3")
    g.add_variable("X2", "L3")
    g.add_variable("X3", "L4")
    g.add_variable("X4", "L4")
    g.add_variable("X5", "L5")
    g.add_variable("X6", "L5")
    g.add_variable("X7", "L1")
    g.add_variable("X8", "L2")
    g.add_variable("X9", "L6")
    g.add_variable("X10", "L6")
    g.add_variable("X11", ["L7", "L8"])
    g.add_variable("X12", ["L7", "L8"])
    g.add_variable("X13", ["L7", "L8"])
    return g



def scenario6():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L1")
    g.add_variable("L6", "L1")
    g.add_variable("L7", "L6")
    g.add_variable("L8", "L6")

    g.add_variable("X1", "L2")
    g.add_variable("X2", "L2")
    g.add_variable("X3", "L2")
    g.add_variable("X4", "L3")
    g.add_variable("X5", ["L3", "L4"])
    g.add_variable("X6", ["L3", "L4"])
    g.add_variable("X7", ["L3", "L4"])
    g.add_variable("X8", "L4")
    g.add_variable("X9", "L1")
    g.add_variable("X10", "L5")
    g.add_variable("X11", "L7")
    g.add_variable("X12", "L7")
    g.add_variable("X13", "L8")
    g.add_variable("X14", "L8")
    return g


def scenario7a():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L2")
    g.add_variable("L5", "L2")
    g.add_variable("L6", "L2")
    g.add_variable("L7", "L3")
    g.add_variable("L8", "L3")
    g.add_variable("L9", "L3")

    g.add_variable("X1", ["L4", "L5", "L6"])
    g.add_variable("X2", ["L4", "L5", "L6"])
    g.add_variable("X3", ["L4", "L5", "L6"])
    g.add_variable("X4", ["L4", "L5", "L6"])
    g.add_variable("X5", ["L4", "L5", "L6"])
    g.add_variable("X6", ["L4", "L5", "L6"])
    g.add_variable("X7", ["L7", "L8", "L9"])
    g.add_variable("X8", ["L7", "L8", "L9"])
    g.add_variable("X9", ["L7", "L8", "L9"])
    g.add_variable("X10", ["L7", "L8", "L9"])
    g.add_variable("X11", ["L7", "L8", "L9"])
    g.add_variable("X12", ["L7", "L8", "L9"])
    g.add_variable("X13", "L1")
    g.add_variable("X14", "L2")
    g.add_variable("X15", "L3")
    return g


def scenario7b():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L2")
    g.add_variable("L5", "L2")
    g.add_variable("L6", "L3")
    g.add_variable("L7", "L3")

    g.add_variable("X1", ["L4", "L5"])
    g.add_variable("X2", ["L4", "L5"])
    g.add_variable("X3", ["L4", "L5"])
    g.add_variable("X4", ["L4", "L5"])
    g.add_variable("X7", ["L6", "L7"])
    g.add_variable("X8", ["L6", "L7"])
    g.add_variable("X9", ["L6", "L7"])
    g.add_variable("X10", ["L6", "L7"])
    g.add_variable("X13", "L2")
    g.add_variable("X14", "L1")
    g.add_variable("X15", "L3")
    return g



def scenario7c():
    g = GaussianGraph()
    g.add_variable("L2", None)
    g.add_variable("L3", "L2")
    g.add_variable("L4", "L2")
    g.add_variable("L5", "L2")
    g.add_variable("L6", "L3")
    g.add_variable("L7", "L3")

    g.add_variable("X1", ["L4", "L5"])
    g.add_variable("X2", ["L4", "L5"])
    g.add_variable("X3", ["L4", "L5"])
    g.add_variable("X4", ["L4", "L5"])
    g.add_variable("X7", ["L6", "L7"])
    g.add_variable("X8", ["L6", "L7"])
    g.add_variable("X9", ["L6", "L7"])
    g.add_variable("X10", ["L6", "L7"])
    g.add_variable("X13", "L2")
    g.add_variable("X14", ["L2", "L3"])
    g.add_variable("X15", "L3")
    return g


def scenario8():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")
    g.add_variable("L4", "L1")
    g.add_variable("L5", "L2")

    g.add_variable("X1", "L3")
    g.add_variable("X2", "L3")
    g.add_variable("X3", "L4")
    g.add_variable("X4", "L4")
    g.add_variable("X5", ["L4", "L5"])
    g.add_variable("X6", ["L4", "L5"])
    g.add_variable("X7", "L2")
    return g



def scenario9a():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")
    g.add_variable("L3", "L1")

    g.add_variable("X1", "L2")
    g.add_variable("X2", "L1")
    g.add_variable("X3", "L3")
    g.add_variable("X4", "L2")
    g.add_variable("X5", "L3")
    g.add_variable("X6", "L3")
    return g



def scenario9b():
    g = GaussianGraph()
    g.add_variable("L1", None)
    g.add_variable("L2", "L1")

    g.add_variable("X1", "L2")
    g.add_variable("X2", ["L1", "L2"])
    g.add_variable("X3", "L1")
    g.add_variable("X4", "L2")
    g.add_variable("X5", "L1")
    g.add_variable("X6", "L1")
    return g

scenarios = {
        "0a": scenario0a,
        "0b": scenario0b,
        "0c": scenario0c,
        "1": scenario1,
        "2": scenario2,
        "3a": scenario3a,
        "3b": scenario3b,
        "3c": scenario3c,
        "4": scenario4,
        "5a": scenario5a,
        "5b": scenario5b,
        "6": scenario6,
        "7a": scenario7a,
        "7b": scenario7b,
        "7c": scenario7c,
        "8": scenario8,
        "9a": scenario9a,
        "9b": scenario9b
        }
