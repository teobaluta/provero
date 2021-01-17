import provero

def test_fixed_intervals():
    # Case where d / alpha < 1
    epsilon = 0.01
    d = 0.2
    alpha = 0.4
    intervals = provero.fixed_intervals(d, alpha, epsilon)
    print('[Case d / alpha < 1 ({}, {}, {})] Intervals: {}'.format(d, alpha,
                                                                   epsilon, intervals))

    # Case where d / alpha < 1
    epsilon = 0.01
    d = 0.2
    alpha = 0.2
    intervals = provero.fixed_intervals(d, alpha, epsilon)
    print('[Case d / alpha < 1 ({}, {}, {})] Intervals: {}'.format(d, alpha,
                                                                   epsilon, intervals))

    # Case where d / alpha < 1
    epsilon = 0.01
    d = 0.8
    alpha = 0.4
    intervals = provero.fixed_intervals(d, alpha, epsilon)
    print('[Case d / alpha < 1 ({}, {},. {})] Intervals: {}'.format(d, alpha,
                                                                    epsilon, intervals))

    # Case where d < epsilon
    epsilon = 0.01
    d = 0.001
    alpha = 0.4
    intervals = provero.fixed_intervals(d, alpha, epsilon)
    print('[Case d < epsilon] Intervals: {}'.format(intervals))
    # ASSERT

def main():
    mu = 0.1
    sigma = 0.1
    d = 0.2
    delta = 0.01

    test_fixed_intervals()
    X = provero.RandomVar(mu, sigma)
    print("Random var normal(mu={},sigma={})".format(mu, sigma))
    # in the non-adaptive search fixing the right alpha seems to be the key
    # if the alpha is too big we will return a None answer - basically we don't
    # know.
    answer = provero.non_adaptive_assert(d, delta, X, timeout=600, alpha=0.01,
                                         epsilon=0.001)
    print("Assert if mu < {} with confidence > {}: {}".format(d, 1 - delta, answer))

if __name__ == "__main__":
    main()
