from scipy import integrate


def p_star(d1, d2, o1):
    return (o1 - d2) / (2 * o1 - d1 - d2)


def EU_opt(d1, d2, o1):
    pstar = p_star(d1, d2, o1)
    return pstar ** 2 * d1 + 2 * pstar * (1 - pstar) * o1 + (1 - pstar) ** 2 * d2


def quality_integrand(d1, d2, o1):
    return 3 * EU_opt(d1, d2, o1)


def vulnerability_integrand(d1, d2, o1):
    return 6 * (EU_opt(d1, d2, o1) - d1)


def check_qualities():
    symmetric_quality, _ = integrate.tplquad(quality_integrand, 0, 1, lambda o1: 0, lambda o1: o1, lambda o1, d2: 0,
                                             lambda o1, d2: o1)
    print("Symmetric quality = {:.3f}".format(symmetric_quality))


def check_vulnerabilities():
    symmetric_vulnerability, _ = integrate.tplquad(vulnerability_integrand, 0, 1, lambda o1: 0, lambda o1: o1,
                                                   lambda o1, d2: 0, lambda o1, d2: d2)
    print("Symmetric vulnerability = {:.3f}".format(symmetric_vulnerability))


if __name__ == "__main__":
    check_qualities()
    check_vulnerabilities()
