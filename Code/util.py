import numpy as np
from scipy.stats import norm

"""
Black Scholes Function

args:
    S = current price of the stock
    K = strike price
    r = the risk-free rate of interest
    v = the annual volatility of the stock's return (can be calculated as the std dev of stock)
    T = the number of years left to expiration
    type "C" for call and "P" for put

local vars:
    PV_K = the present value of a risk-free zero-coupon bond that pays K on option expiration date (K is the exercise price)
    d1, d2 = probability factors
    
output:
    price = option value
"""

# The following function works on non-dividend-paying American or European call/put options
def blackScholes(S, K, r, v, T, type):
    PV_K = K/np.exp(r*T)
    d1 = (np.log(S/PV_K))/(v*np.sqrt(T)) + ((v*np.sqrt(T))/2)
    d2 = d1 - v*np.sqrt(T)
    
    if type == "C":
        price = S * norm.cdf(d1, 0, 1) - PV_K * norm.cdf(d2, 0, 1)
    elif type == "P":
        price = (PV_K * norm.cdf(-d2, 0, 1)) - (S * norm.cdf(-d1, 0, 1))
    
    return price

"""
Binomial Options Pricing Function (utilizing backpropagation)

args:
    T = the number of years left to expiration
    K = the strike price
    S_init = the initial stock price
    r = the risk-free rate of interest (in hundredths)
    N = # of time steps
    u = up-state factor
    d = 1/u --> down-state factor
    sigma = implied volatility
    type "C" for call and "P" for put

local vars:
    rn_p = the risk-neutral probability
    discount = the factor that involves the compounded interest --> rate of 1+r OR e^rT

output:
    C[0] = option value
"""
# TODO: implement functions for dividend paying stocks

# It is never optimal to exercise American calls early, so they behave like European calls
# USER GUIDE: this works on American puts/American calls/European calls for non-dividend paying stocks
def binomial_tree_simple(T, K, S_init, r, N, u, d, type):
    dt = T/N
    rnp = ((1+r)*S_init-(S_init*d))/(S_init*u-S_init*d) 
    discount = 1+r

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            C[j] = (rnp*C[j+1] + (1-rnp)*C[j])/discount
            if type == "P":
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)
                
    return C[0]

# USER GUIDE: this works on American puts/American calls/European calls 
def binomial_tree_compounded(T, K, S_init, r, N, u, d, type):
    dt = T/N
    rnp = (np.exp(r*dt) - d)/(u-d)
    discount = np.exp(r*dt)

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            C[j] = (rnp*C[j+1] + (1-rnp)*C[j])/discount
            if type == "P":
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)
                
    return C[0]

# USER GUIDE: this works on American puts/American calls/European calls (CRR METHOD)
def binomial_tree_compounded_CRR(T, K, S_init, r, N, sigma, type):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp(r*dt) - d)/(u-d)
    discount = np.exp(r*dt)

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            C[j] = (rnp*C[j+1] + (1-rnp)*C[j])/discount
            if type == "P":
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)
                
    return C[0]

# USER GUIDE: this works on American puts/American calls/European calls (JR METHOD)
def binomial_tree_compounded_JR(T, K, S_init, r, N, sigma, type):
    dt = T/N
    nu = r - 0.5*sigma**2
    u = np.exp(nu*dt + sigma*np.sqrt(dt))
    d = np.exp(nu*dt - sigma*np.sqrt(dt))
    rnp = 0.5
    discount = np.exp(r*dt)

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            C[j] = (rnp*C[j+1] + (1-rnp)*C[j])/discount
            if type == "P":
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)
                
    return C[0]

# USER GUIDE: this works on European puts
def binomial_tree_euro_put(T, K, S_init, r, N, u, d):
    dt = T / N
    rnp = ((1 + r) * S_init - (S_init * d)) / (S_init * u - S_init * d)
    discount = 1 + r
    #rnp = (np.exp(r*dt) - d)/(u-d)
    #discount = np.exp(r*dt)

    S = np.zeros(N + 1)
    for j in range(0, N + 1):
        S[j] = S_init * (u ** j) * (d ** (N - j))

    C = np.zeros(N + 1)
    for j in range(0, N + 1):
        C[j] = max(0, K - S[j])

    for i in np.arange(N - 1, -1, -1):
        for j in range(0, i + 1):
            C[j] = (rnp * C[j + 1] + (1 - rnp) * C[j]) / discount

    return C[0]

# USER GUIDE: this works on American puts/American calls/European calls 
def expected_payoff_tree_simple(T, K, S_init, r, N, u, d, type):
    """
    E is the expected payoff matrix
    """
    dt = T/N
    rnp = ((1+r)*S_init-(S_init*d))/(S_init*u-S_init*d) 

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    E = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            E[j] = max(0, K - S[j])
        else:
            E[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            E[j] = rnp*E[j+1] + (1-rnp)*E[j]
            if type == "P":
                E[j] = max(E[j], K - S)
            else:
                E[j] = max(E[j], S - K)
                
    return E[0]

# USER GUIDE: this works on American puts/American calls/European calls 
def expected_payoff_tree_compounded(T, K, S_init, r, N, u, d, type):
    """
    E is the expected payoff matrix
    """
    dt = T/N
    rnp = (np.exp(r*dt) - d)/(u-d)

    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S_init * u**j * d**(N-j)
        
    E = np.zeros(N+1)
    for j in range(0, N+1):
        if type == "P":
            E[j] = max(0, K - S[j])
        else:
            E[j] = max(0, S[j] - K)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S_init * u**j * d**(i-j)
            E[j] = rnp*E[j+1] + (1-rnp)*E[j]
            if type == "P":
                E[j] = max(E[j], K - S)
            else:
                E[j] = max(E[j], S - K)
                
    return E[0]

# USER GUIDE: this works on European puts
def expected_payoff_tree_euro_put(T, K, S_init, r, N, u, d):
    """
    E is the expected payoff matrix
    """
    dt = T / N
    rnp = ((1 + r) * S_init - (S_init * d)) / (S_init * u - S_init * d)
    discount = 1 + r
    #rnp = (np.exp(r*dt) - d)/(u-d)
    #discount = np.exp(r*dt)

    S = np.zeros(N + 1)
    for j in range(0, N + 1):
        S[j] = S_init * (u ** j) * (d ** (N - j))

    E = np.zeros(N + 1)
    for j in range(0, N + 1):
        E[j] = max(0, K - S[j])

    for i in np.arange(N - 1, -1, -1):
        for j in range(0, i + 1):
            E[j] = (rnp * E[j + 1] + (1 - rnp) * E[j])

    return E[0]

# Calculating up state and down state from volatility & compounded interest rates for non dividend-paying stocks --> (up, down)
def find_states(T, N, r, sigma):
    dt = T/N
    u = np.exp((r*dt) + sigma * np.sqrt(dt))
    d = np.exp((r*dt) - sigma * np.sqrt(dt))

    return u, d

    dt = T/N
    u = 1 + r + sigma*np.sqrt(dt)
    d = 1 + r - sigma*np.sqrt(dt)

    return u, d


"""
Testing
"""
# print(binomial_tree_compounded(2, 23, 20, 0.05, 2, 1.3, 0.9, type="C"))
# print(binomial_tree_euro_put(0.5, 100, 102, 0.02, 4, 1.0733, 0.9317))
