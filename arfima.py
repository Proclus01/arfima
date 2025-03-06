#!/usr/bin/env python3
"""ARFIMA Library

This module implements an autoregressive fractionally integrated moving average (ARFIMA)
model. The model is expressed as

    (1 - Σ_{i=1}^p φᵢ Bᶦ) (1-B)^d Xₜ = (1 + Σ_{j=1}^q θⱼ Bⱼ) εₜ ,

where d may be fractional so that the differencing operator is defined via a binomial
expansion. Fractional differencing/integration is done by computing the appropriate
weights (with an adaptive truncation based on a threshold).

The ARFIMA class includes methods to:
    • Simulate data from an ARFIMA model.
    • Fit the model's parameters (d, AR coefficients, MA coefficients) via minimization of
      the sum of squared innovations (a conditional least-squares style objective).
    • Forecast future values by forecasting in the “stationary” (fractionally differenced)
      domain and then applying the inverse fractional differencing operator.

All functions and methods are fully type-annotated.
"""

import numpy as np
from typing import Optional, List
from scipy.optimize import minimize


def get_frac_diff_weights(d: float, thresh: float = 1e-5) -> np.ndarray:
    """
    Compute the fractional differencing weights for a given differencing parameter d.
    The weights are computed iteratively until the absolute value of the next coefficient
    drops below a threshold.
    
    Args:
        d: The fractional differencing parameter.
        thresh: The threshold for truncation of the infinite series.
        
    Returns:
        A 1D numpy array of weights.
    """
    weights: List[float] = [1.0]
    k: int = 1
    while True:
        w: float = weights[-1] * ((d - k + 1) / k) * -1
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
    return np.array(weights, dtype=float)


def fractional_difference(series: np.ndarray, d: float, thresh: float = 1e-5) -> np.ndarray:
    """
    Apply fractional differencing to a series.
    
    That is, compute Yₜ = (1-B)^d Xₜ = Σₖ gₖ Xₜ₋ₖ, where the weights gₖ are computed via
    a binomial expansion with truncation defined by the threshold.
    
    Args:
        series: The original time series, X.
        d: The fractional differencing parameter.
        thresh: The truncation threshold for the weights.
        
    Returns:
        The fractionally differenced series Y.
    """
    weights: np.ndarray = get_frac_diff_weights(d, thresh)
    n: int = len(series)
    frac_diff: np.ndarray = np.zeros(n, dtype=float)
    Nw: int = len(weights)
    for t in range(n):
        tmp: float = 0.0
        # Only include as many weights as available data (using zero padding for t-k < 0)
        for k in range(min(Nw, t + 1)):
            tmp += weights[k] * series[t - k]
        frac_diff[t] = tmp
    return frac_diff


def compute_innovations(Y: np.ndarray, ar_params: np.ndarray, ma_params: np.ndarray) -> np.ndarray:
    """
    Given a stationary series Y (obtained by fractional differencing the original X),
    and AR/MA parameters (for the finite ARMA representation), compute the one-step
    ahead residual innovations via the recursion:
    
      εₜ = Yₜ - Σᵢ φᵢ Yₜ₋ᵢ - Σⱼ θⱼ εₜ₋ⱼ.
    
    For t where lags are not available, zero values are assumed.
    
    Args:
        Y: The stationary series (after fractional differencing).
        ar_params: The AR coefficients (if any) as a 1D numpy array.
        ma_params: The MA coefficients (if any) as a 1D numpy array.
        
    Returns:
        A numpy array of computed innovations ε.
    """
    T: int = len(Y)
    eps: np.ndarray = np.zeros(T, dtype=float)
    p: int = len(ar_params)
    q: int = len(ma_params)
    for t in range(T):
        ar_sum: float = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                ar_sum += ar_params[i - 1] * Y[t - i]
        ma_sum: float = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                ma_sum += ma_params[j - 1] * eps[t - j]
        eps[t] = Y[t] - ar_sum - ma_sum
    return eps


class ARFIMA:
    """
    A fully type-annotated, object-oriented implementation of an ARFIMA model.
    
    The model is defined via:
    
       (1 - Σ₁^p φᵢBᶦ)(1-B)^d Xₜ = (1 + Σ₁^q θⱼBʲ)εₜ.
    
    Once initialized with orders (and optionally initial parameter guesses) you can
    simulate data, fit the model to data, or forecast future values.
    """
    def __init__(
        self,
        d: float = 0.0,
        ar_params: Optional[List[float]] = None,
        ma_params: Optional[List[float]] = None
    ) -> None:
        """
        Initialize the ARFIMA model.
        
        Args:
            d: The fractional differencing parameter.
            ar_params: A list of AR coefficients φ. (Default is an empty list.)
            ma_params: A list of MA coefficients θ. (Default is an empty list.)
        """
        self.d: float = d
        self.ar: np.ndarray = np.array(ar_params, dtype=float) if ar_params is not None else np.array([])
        self.ma: np.ndarray = np.array(ma_params, dtype=float) if ma_params is not None else np.array([])

    def fit(self, data: np.ndarray, start_params: Optional[np.ndarray] = None, thresh: float = 1e-5) -> None:
        """
        Fit the ARFIMA model to the time series data.
        
        This method estimates the differencing parameter d, along with any AR or MA
        parameters, by minimizing the sum of squared residuals computed via the ARMA recursion.
        
        Args:
            data: The original time series (in its original units).
            start_params: Optional initial guess for the parameters in the vector form 
                          [d, (AR parameters), (MA parameters)]. If not provided,
                          the current self.d, self.ar, and self.ma are used.
            thresh: The threshold for truncation in computing fractional weights.
        """
        p: int = len(self.ar)
        q: int = len(self.ma)
        
        def objective(params: np.ndarray) -> float:
            candidate_d: float = params[0]
            candidate_ar: np.ndarray = params[1:1+p] if p > 0 else np.array([])
            candidate_ma: np.ndarray = params[1+p:1+p+q] if q > 0 else np.array([])
            # Fractionally difference the data with the candidate d:
            Y: np.ndarray = fractional_difference(data, candidate_d, thresh=thresh)
            eps: np.ndarray = compute_innovations(Y, candidate_ar, candidate_ma)
            # Under a Gaussian error assumption, minimizing the SSE is equivalent to ML estimation.
            return np.sum(eps ** 2)
        
        # Assemble initial guess
        if start_params is None:
            init_params: np.ndarray = np.array(
                [self.d] + (self.ar.tolist() if p > 0 else []) + (self.ma.tolist() if q > 0 else []),
                dtype=float
            )
        else:
            init_params = start_params
        
        res = minimize(objective, init_params, method="Nelder-Mead")
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        optimized: np.ndarray = res.x
        self.d = optimized[0]
        if p > 0:
            self.ar = optimized[1:1+p]
        if q > 0:
            self.ma = optimized[1+p:1+p+q]

    def predict(self, data: np.ndarray, steps: int, thresh: float = 1e-5) -> np.ndarray:
        """
        Forecast future values from the ARFIMA model.
        
        The forecast is computed in two stages. First, the original series
        is fractionally differenced to obtain a stationary series Y. Then an ARMA forecast
        for Y is computed (by using the AR part and assuming future errors vanish).
        Finally, the inverse fractional differencing operator (the fractional integration)
        is applied to obtain forecasts in the original scale.
        
        Args:
            data: The original, observed time series.
            steps: The number of future steps to forecast.
            thresh: The threshold for truncation in computing weights.
        
        Returns:
            A numpy array of forecasted values (in the same units as data).
        """
        # Compute fractional differencing of data
        Y: np.ndarray = fractional_difference(data, self.d, thresh=thresh)
        T: int = len(Y)
        p: int = len(self.ar)
        q: int = len(self.ma)
        
        # Compute innovations for the in-sample part
        eps: np.ndarray = compute_innovations(Y, self.ar, self.ma)
        
        # Forecast the stationary series Y using the AR part.
        # We assume that future innovations (and hence MA contributions) are zero.
        forecast_Y: np.ndarray = np.zeros(steps, dtype=float)
        # Build an extended list of Y values (starting with observed ones)
        Y_extended: List[float] = Y.tolist()
        for h in range(steps):
            ar_sum: float = 0.0
            for i in range(1, p + 1):
                # For forecast lags use the most recent observed or forecasted values
                if h - i < 0:
                    ar_sum += self.ar[i - 1] * Y_extended[T + h - i]
                else:
                    ar_sum += self.ar[i - 1] * forecast_Y[h - i]
            forecast_Y[h] = ar_sum
            Y_extended.append(ar_sum)
        
        # Invert the fractional differencing operator by applying fractional integration.
        # (1-B)^{-d} can be expanded as ψ(B) where ψ₀ = 1 and for k>=1:
        #    ψₖ = ψₖ₋₁ * ((d + k - 1) / k)
        psi: List[float] = [1.0]
        k: int = 1
        while True:
            psi_val: float = psi[-1] * ((self.d + k - 1) / k)
            if abs(psi_val) < thresh:
                break
            psi.append(psi_val)
            k += 1
        psi_arr: np.ndarray = np.array(psi, dtype=float)
        
        # Forecast X (the original series) from the forecast Y.
        # Note: One way is to reapply the integration relationship:
        #   Xₜ = Σ_{j=0}^t ψⱼ Yₜ₋ⱼ.
        X_forecast: np.ndarray = np.zeros(steps, dtype=float)
        # In this simple approach we use the forecasted Y values along with the integration weights
        # and assume that only the forecasted (future) Y contribute for h>=0.
        for h in range(steps):
            summation: float = 0.0
            # We sum over all available lags (observed plus forecasted)
            for j in range(min(len(psi_arr), T + h + 1)):
                if T + h - j < len(Y_extended):
                    summation += psi_arr[j] * Y_extended[T + h - j]
            X_forecast[h] = summation
        return X_forecast

    def simulate(
        self, n: int, burn: int = 1000, thresh: float = 1e-5, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate a time series from the ARFIMA model.
        
        The simulation is done by:
          1. Generating an underlying ARMA process Yₜ via
              Yₜ = Σᵢ φᵢ Yₜ₋ᵢ + εₜ + Σⱼ θⱼ εₜ₋ⱼ
             (with white noise εₜ drawn from N(0,1)),
          2. Then applying the inverse fractional differencing operator: 
              Xₜ = (1-B)^{-d} Yₜ.
        
        Args:
            n: The desired length of the simulated series.
            burn: The number of initial samples to discard (to reduce initialization effects).
            thresh: The threshold for truncation in computing fractional weights.
            random_state: An optional seed for the random number generator.
        
        Returns:
            A simulated time series of length n.
        """
        if random_state is not None:
            np.random.seed(random_state)
        total: int = n + burn
        eps: np.ndarray = np.random.normal(0, 1, total)
        Y: np.ndarray = np.zeros(total, dtype=float)
        p: int = len(self.ar)
        q: int = len(self.ma)
        
        # Simulate the ARMA process for Yₜ.
        for t in range(total):
            ar_sum: float = 0.0
            for i in range(1, p + 1):
                if t - i >= 0:
                    ar_sum += self.ar[i - 1] * Y[t - i]
            ma_sum: float = 0.0
            for j in range(1, q + 1):
                if t - j >= 0:
                    ma_sum += self.ma[j - 1] * eps[t - j]
            Y[t] = ar_sum + eps[t] + ma_sum
        
        # Compute the integration (inverse differencing) weights.
        psi: List[float] = [1.0]
        k: int = 1
        while True:
            psi_val: float = psi[-1] * ((self.d + k - 1) / k)
            if abs(psi_val) < thresh:
                break
            psi.append(psi_val)
            k += 1
        psi_arr: np.ndarray = np.array(psi, dtype=float)
        N_psi: int = len(psi_arr)
        X: np.ndarray = np.zeros(total, dtype=float)
        
        # Apply the fractional integration: Xₜ = Σₖ ψₖ Yₜ₋ₖ.
        for t in range(total):
            summation: float = 0.0
            for k in range(min(N_psi, t + 1)):
                summation += psi_arr[k] * Y[t - k]
            X[t] = summation
        
        # Return the simulated series discarding burn-in.
        return X[burn:]


# Example usage (if run as a script)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1. Simulate an ARFIMA(0, d, 0) process with d = 0.3.
    model = ARFIMA(d=0.3)
    simulated_series: np.ndarray = model.simulate(n=500, random_state=42)
    plt.figure()
    plt.plot(simulated_series)
    plt.title("Simulated ARFIMA(0, 0.3, 0) Series")
    plt.xlabel("Time")
    plt.ylabel("Xₜ")
    plt.show()
    
    # Example 2. Fit an ARFIMA model to the simulated data.
    # Here we start with an initial guess for d (and no AR or MA terms).
    model_fit = ARFIMA(d=0.2)
    model_fit.fit(simulated_series)
    print("Estimated d:", model_fit.d)
    
    # Forecast the next 10 values.
    forecasts: np.ndarray = model_fit.predict(simulated_series, steps=10)
    print("Forecasts:", forecasts)
    
    plt.figure()
    plt.plot(np.arange(len(simulated_series)), simulated_series, label="Observed")
    plt.plot(
        np.arange(len(simulated_series), len(simulated_series) + 10),
        forecasts,
        label="Forecast",
        marker='o'
    )
    plt.xlabel("Time")
    plt.legend()
    plt.title("ARFIMA Forecast")
    plt.show()