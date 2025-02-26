import bambi as bmb
from neonormalcb.distributions import fsst, msnburr, msnburr_iia

print("fsst:", fsst)
print("msnburr:", msnburr)
print("msnburr_iia:", msnburr_iia)

# FSST Family
fsst_likelihood = bmb.Likelihood(
    "fsst",
    params = ["mu", "sigma", "nu", "alpha"],
    parent = "mu",
    dist = fsst.dist
)
fsst_links = {"mu": "identity", "sigma": "log", "nu": "log", "alpha": "log"}
fsst_family = bmb.Family("fsst", fsst_likelihood, fsst_links)

# MSNBurr Family
msnburr_likelihood = bmb.Likelihood(
    "msnburr",
    params = ["mu", "sigma", "alpha"],
    parent ="mu",
    dist = msnburr.dist
)
msnburr_links = {"mu": "identity", "sigma": "log", "alpha": "log"}
msnburr_family = bmb.Family("msnburr", msnburr_likelihood, msnburr_links)

# MSNBurr-IIa Family
msnburr_iia_likelihood = bmb.Likelihood(
    "msnburr_iia",
    params = ["mu", "sigma", "alpha"],
    parent = "mu",
    dist = msnburr_iia.dist
)
msnburr_iia_links = {"mu": "identity", "sigma": "log", "alpha": "log"}
msnburr_iia_family = bmb.Family("msnburr_iia", msnburr_iia_likelihood, msnburr_iia_links)
