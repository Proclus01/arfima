# test_arfima.py
import numpy as np
import pytest
from arfima import (
    get_frac_diff_weights,
    fractional_difference,
    compute_innovations,
    ARFIMA,
)


def test_frac_diff_weights():
    # When d = 0, the weights should be [1.0]
    d = 0.0
    weights = get_frac_diff_weights(d)
    np.testing.assert_array_almost_equal(weights, np.array([1.0]))

    # Test for a fractional differencing parameter d = 0.5.
    # Expected weights for the first few terms (computed iteratively):
    # weight[0] = 1.0
    # weight[1] = -0.5
    # weight[2] = -0.125
    # weight[3] = -0.0625
    d = 0.5
    weights = get_frac_diff_weights(d, thresh=1e-8)
    expected_first_four = np.array([1.0, -0.5, -0.125, -0.0625])
    np.testing.assert_allclose(weights[:4], expected_first_four, rtol=1e-5)


def test_fractional_difference():
    # Create a simple increasing series.
    series = np.arange(1, 11, dtype=float)  # [1,2,...,10]
    d = 0.5
    frac_diff_series = fractional_difference(series, d, thresh=1e-8)
    # Ensure the output has the same shape as the input.
    assert frac_diff_series.shape == series.shape
    # Check that every computed value is finite.
    assert np.all(np.isfinite(frac_diff_series))


def test_compute_innovations():
    # Using a simple series where no AR or MA exists: innovations equal Y.
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    eps = compute_innovations(Y, np.array([]), np.array([]))
    np.testing.assert_array_almost_equal(eps, Y)

    # Test with one AR parameter.
    # With AR(1) parameter φ = 0.5, for example:
    #   ε[0] = Y[0]
    #   ε[1] = Y[1] - 0.5 * Y[0]
    #   ε[2] = Y[2] - 0.5 * Y[1]
    ar_params = np.array([0.5])
    eps_ar = compute_innovations(Y, ar_params, np.array([]))
    expected = np.array([1.0, 2.0 - 0.5 * 1.0, 3.0 - 0.5 * 2.0, 4.0 - 0.5 * 3.0, 5.0 - 0.5 * 4.0])
    np.testing.assert_array_almost_equal(eps_ar, expected)


def test_simulate_and_predict():
    # Use ARFIMA model with d = 0.3 and zero AR/MA.
    model = ARFIMA(d=0.3)
    simulated_series = model.simulate(n=200, random_state=123)
    # Ensure the length of the simulated series is correct.
    assert simulated_series.shape[0] == 200
    # Check that the series is not trivially zero.
    assert np.any(simulated_series != 0)

    # Test the forecast method
    forecast_steps = 5
    forecasts = model.predict(simulated_series, steps=forecast_steps)
    assert forecasts.shape[0] == forecast_steps
    # Ensure forecast values are finite.
    assert np.all(np.isfinite(forecasts))


def test_fit():
    # Simulate data using a known d value.
    true_d = 0.4
    model_true = ARFIMA(d=true_d)
    simulated_series = model_true.simulate(n=300, random_state=456)

    # Create a new ARFIMA instance with an initial guess different than the truth.
    model_fit_instance = ARFIMA(d=0.2)
    model_fit_instance.fit(simulated_series)

    # Check that the estimated d is within 0.15 of the true d.
    estimation_error = abs(model_fit_instance.d - true_d)
    assert estimation_error < 0.15, f"Estimated d: {model_fit_instance.d}, True d: {true_d}"


if __name__ == "__main__":
    pytest.main(["-v"])