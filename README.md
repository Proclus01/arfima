# ARFIMA Library

This library implements fractional differencing and integration using a binomial expansion and supports simulation, fitting, and forecasting for ARFIMA models.

## Overview

ARFIMA models generalize ARIMA processes by allowing the differencing parameter to take non-integer (fractional) values. They are particularly useful for modeling time series with long memory characteristics. This library provides:

- **Fractional Differencing/Integration:** Computes the fractional differencing weights with adaptive truncation.
- **ARMA Innovations Recursion:** Calculates one-step-ahead residuals for the fractionally differenced (stationary) series using AR and MA components.
- **Simulation:** Generates synthetic time series data from ARFIMA models.
- **Parameter Estimation:** Fits model parameters (d, AR, MA) by minimizing the sum of squared innovations.
- **Forecasting:** Makes predictions using the fitted model by forecasting in the stationary domain, then applying fractional integration to revert to the original scale.

## Features

- Fully type-annotated code for improved clarity and maintainability.
- Object-oriented design with an `ARFIMA` class that encapsulates model functionality.
- Implements correct numerical methods for fractional differencing/integration.
- Example usage for simulating data, fitting a model, and generating forecasts.

## Requirements

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/) (for plotting in the examples)

You can install NumPy, SciPy, and Matplotlib via pip if they are not already installed:

```bash
pip install numpy scipy matplotlib
```

## Installation

Since the ARFIMA library is built from scratch and is provided as a single module, you can simply download the `arfima.py` file and include it in your Python project. Alternatively, if you plan on modifying or extending the library, consider cloning the repository.

## Usage

Below is an example demonstrating how to use the ARFIMA library:

1. **Simulation:** Generate a synthetic ARFIMA(0, d, 0) time series.
2. **Fitting:** Estimate the fractional differencing parameter (d) from the simulated series.
3. **Forecasting:** Forecast future values based on the fitted model.

```python
import numpy as np
import matplotlib.pyplot as plt
from arfima import ARFIMA

# 1. Simulate an ARFIMA(0, d, 0) process with d = 0.3
model_sim = ARFIMA(d=0.3)
simulated_series = model_sim.simulate(n=500, random_state=42)

plt.figure()
plt.plot(simulated_series)
plt.title("Simulated ARFIMA(0, 0.3, 0) Series")
plt.xlabel("Time")
plt.ylabel("Xₜ")
plt.show()

# 2. Fit an ARFIMA model to the simulated data.
# Start with an initial guess for d (here, 0.2) and no AR or MA components.
model_fit = ARFIMA(d=0.2)
model_fit.fit(simulated_series)
print("Estimated d:", model_fit.d)

# 3. Forecast the next 10 values.
forecasts = model_fit.predict(simulated_series, steps=10)
print("Forecasts:", forecasts)

plt.figure()
plt.plot(np.arange(len(simulated_series)), simulated_series, label="Observed")
plt.plot(np.arange(len(simulated_series), len(simulated_series) + 10),
         forecasts, label="Forecast", marker='o')
plt.xlabel("Time")
plt.legend()
plt.title("ARFIMA Forecast")
plt.show()
```

### API Documentation

- **`get_frac_diff_weights(d: float, thresh: float = 1e-5) -> np.ndarray`**  
  Computes fractional differencing weights for a given parameter d using adaptive truncation.

- **`fractional_difference(series: np.ndarray, d: float, thresh: float = 1e-5) -> np.ndarray`**  
  Applies fractional differencing to the provided time series.

- **`compute_innovations(Y: np.ndarray, ar_params: np.ndarray, ma_params: np.ndarray) -> np.ndarray`**  
  Computes one-step-ahead innovations using the AR and MA parameters for a stationary series.

- **`class ARFIMA`**  
  The main class implementing the ARFIMA model with methods:
  - `simulate(n: int, burn: int = 1000, thresh: float = 1e-5, random_state: Optional[int] = None) -> np.ndarray`
  - `fit(data: np.ndarray, start_params: Optional[np.ndarray] = None, thresh: float = 1e-5) -> None`
  - `predict(data: np.ndarray, steps: int, thresh: float = 1e-5) -> np.ndarray`

For a complete understanding and for making modifications, please refer to the source code and inline documentation.

## Contributing

Contributions to the ARFIMA library are welcome! If you have ideas for improvements or bug fixes, please fork the repository and submit a pull request. Enhancements to the numerical methods, optimization routines, or additional features are highly encouraged.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) file for details.

## Acknowledgments

This library is inspired by the ARFIMA model formulations presented in statistical literature, and it aims to provide a solid foundation for ARFIMA modeling in Python.