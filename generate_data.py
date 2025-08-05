import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

def generate_patient_csv(n_patients, delta, T, mu=12, std=5.0, folder=".", filename="patients.csv"):
    """
    Generate patient CSV in the specified folder (default: current),
    using normal distribution for number of sessions (d_p ~ N(mu, std)).
    Discharge time (nu_p) is fixed and known; sigma_p represents uncertainty in session need.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    patients = []

    for p in range(n_patients):
        d_p = int(np.clip(np.random.normal(loc=mu, scale=std), 2, delta - 1))  # sessions needed
        sigma_p = np.random.uniform(0.5, 2.0)  # std dev for session prediction uncertainty
        nu_p = np.random.randint(15, len(T))    # fixed known discharge slot

        patients.append({
            "patient_id": p,
            "d_p": d_p,
            "nu_p": nu_p,
            "sigma_p": sigma_p
        })

    df = pd.DataFrame(patients)
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")
    return path


def load_patients_from_csv(path, T, delta):
    df = pd.read_csv(path)
    P = df["patient_id"].tolist()
    d = {row["patient_id"]: int(row["d_p"]) for _, row in df.iterrows()}
    nu = {row["patient_id"]: row["nu_p"] for _, row in df.iterrows()}
    sigma_std = {row["patient_id"]: row["sigma_p"] for _, row in df.iterrows()}

    sigma = {
        p: int(np.clip(np.random.normal(nu[p], sigma_std[p]), 3, len(T)))
        for p in P
    }

    return P, d, nu, sigma_std, sigma