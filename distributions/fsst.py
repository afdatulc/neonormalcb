import pymc as pm
import numpy as np
import scipy
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
from pymc.distributions.dist_math import check_parameters

def logp(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, nu: TensorVariable, alpha: TensorVariable):
    """ Log-probability function for ST3 distribution """
    sigma = pt.as_tensor_variable(sigma).astype('floatX')
    alpha = pt.as_tensor_variable(alpha).astype('floatX')
    nu = pt.as_tensor_variable(nu).astype('floatX')

    # Student-T components (loglik_a)
    loglik1a = pm.logp(pm.StudentT.dist(nu), (alpha * (y - mu) / sigma))
    loglik2a = pm.logp(pm.StudentT.dist(nu), ((y - mu) / (sigma * alpha)))

    loglika = pt.switch(y < mu, loglik1a, loglik2a)
    loglika += pt.log(2 * alpha / (1 + alpha**2)) - pt.log(sigma)

    # Normal approximation (loglik_b)
    loglik1b = pm.logp(pm.Normal.dist(0, 1), (alpha * (y - mu) / sigma))
    loglik2b = pm.logp(pm.Normal.dist(0, 1), ((y - mu) / (sigma * alpha)))

    loglikb = pt.switch(y < mu, loglik1b, loglik2b)
    loglikb += pt.log(2 * alpha / (1 + alpha**2)) - pt.log(sigma)

    # Switch between Student-T and Normal based on nu threshold
    threshold_nu = 1e6
    logp = pt.switch(nu < threshold_nu, loglika, loglikb)

    return check_parameters(
        logp,
        sigma > 0,
        nu > 0,
        alpha > 0,
        msg=f"sigma, nu, and alpha must be positive"
    )

def logcdf(y: TensorVariable, mu: TensorVariable, sigma: TensorVariable, nu: TensorVariable, alpha: TensorVariable, **kwargs):
    """ Log-cumulative function for ST3 distribution """

    logcdf1 = (2 / (1 + alpha**2)) * pt.exp(pm.logcdf(pm.StudentT.dist(nu), (alpha * (y - mu) / sigma)))
    logcdf2 = (1 / (1 + alpha**2)) * (1 + 2 * alpha**2 *
                                  (pt.exp(pm.logcdf(pm.StudentT.dist(nu), (y - mu) / (sigma * alpha))) - 0.5))

    logcdf = pt.switch(y < mu, logcdf1, logcdf2)
    logcdf = pt.log(logcdf)

    return check_parameters(
        logcdf,
        sigma > 0,
        nu > 0,
        alpha > 0,
        msg=f"sigma, nu, and alpha must be positive"
    )

def quantile(p: TensorVariable, mu: TensorVariable, sigma: TensorVariable, nu: TensorVariable, alpha: TensorVariable):
    """Quantile function (inverse CDF) for FSST distribution"""

    q1 = mu + (sigma / alpha) * pm.icdf(pm.StudentT.dist(nu), p * (1 + alpha**2) / 2)
    q2 = mu + (sigma * alpha) * pm.icdf(pm.StudentT.dist(nu), ((p * (1 + alpha**2) - 1) / (2 * alpha**2)) + 0.5)

    # Menyesuaikan hasil seperti qST3 di R
    q = pt.switch(p < (1 / (1 + alpha**2)), q1, q2)

    # Menangani batas p = 0 dan p = 1
    q = pt.switch(pt.eq(p, 0), -np.inf, q)
    q = pt.switch(pt.eq(p, 1), np.inf, q)


    return q


def random(
      mu: np.ndarray | float,
      sigma: np.ndarray | float,
      nu: np.ndarray | float,
      alpha: np.ndarray | float,
      rng = np.random.default_rng(),
      # size: Optional[int | Tuple[int, ...]] = None,
      size: Optional[Tuple[int]]=None,
    ):

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    size = size or ()

    # if rng is None:
    #   rng = np.random.default_rng()

    u = rng.uniform(low=0, high=1, size=size)
    # q1 = mu + (sigma / alpha) * pm.icdf(pm.StudentT.dist(nu), u * (1 + alpha**2) / 2)
    # q2 = mu + (sigma * alpha) * pm.icdf(pm.StudentT.dist(nu), ((u * (1 + alpha**2) - 1) / (2 * alpha**2)) + 0.5)

    # random = quantile(u, mu, sigma, nu, alpha)

    q1 = mu + (sigma / alpha) * scipy.stats.t.ppf(u * (1 + alpha**2) / 2, df=nu)
    q2 = mu + (sigma * alpha) * scipy.stats.t.ppf(((u * (1 + alpha**2) - 1) / (2 * alpha**2)) + 0.5, df=nu)

    # u_np = np.asarray(u)  # Konversi tensor uniform menjadi array NumPy
    # q1_np = np.asarray(q1)
    # q2_np = np.asarray(q2)
    # q = np.where(u_np < (1 / (1 + alpha**2)), q1_np, q2_np)

    q = np.where(u < (1 / (1 + alpha**2)), q1, q2)

    return np.asarray(q)


class fsst:
    def __new__(self, name:str, mu, sigma, nu, alpha, observed=None, **kwargs):
        return pm.CustomDist(
            name,
            mu, sigma, nu, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            observed=observed,
            **kwargs
        )

    @classmethod
    def dist(cls, mu, sigma, nu, alpha, **kwargs):
        return pm.CustomDist.dist(
            mu, sigma, nu, alpha,
            logp=logp,
            logcdf=logcdf,
            random=random,
            # moment=moment
        )


    @staticmethod
    def icdf(p, mu, sigma, nu, alpha):
        return quantile(p, mu, sigma, nu, alpha)
