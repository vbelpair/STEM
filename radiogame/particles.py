import numpy as np
from dataclasses import dataclass
from typing import List

def _weights_to_int_dose(weights: np.ndarray, E: int) -> List[int]:
    """Convert positive weights to integer doses summing exactly to E."""
    E = int(max(0, E))
    if E == 0:
        return []

    w = np.maximum(weights.astype(float), 0.0)
    if w.sum() <= 0:
        return [E]

    scaled = w / w.sum() * E
    base = np.floor(scaled).astype(int)
    rem = scaled - base
    missing = E - int(base.sum())

    if missing > 0:
        idx = np.argsort(-rem)[:missing]
        base[idx] += 1

    # ensure exact sum
    diff = E - int(base.sum())
    if diff != 0:
        base[0] += diff

    return base.tolist()


@dataclass(frozen=True)
class Particle:
    name: str

    def deposits(self, E: int, n_steps: int) -> List[int]:
        raise NotImplementedError


class Proton(Particle):
    """
    Proton: low entrance, Bragg peak near the end of its range,
    then zeros afterwards (still traveling but no more dose).
    """
    def __init__(self):
        super().__init__(name="proton")

    def deposits(self, E: int, n_steps: int) -> List[int]:
        E = int(max(1, E))
        n_steps = int(max(1, n_steps))

        # Range in tiles (how long it meaningfully deposits) – tunable
        R = max(2, int(round(0.6 * E)) + 2)
        R = min(R, n_steps)

        i = np.arange(R, dtype=float)
        x = i / max(1, R - 1)

        # baseline ramp + sharp peak at end
        baseline = 0.35 + 0.65 * x
        peak = 3.8 * np.exp(-0.5 * ((x - 1.0) / 0.12) ** 2)
        weights = baseline + peak

        dose_R = _weights_to_int_dose(weights, E)

        # pad with zeros AFTER range
        out = dose_R + [0] * (n_steps - R)
        return out


class Electron(Particle):
    """
    Electron: very local; deposits quickly and then effectively stops (zeros).
    """
    def __init__(self):
        super().__init__(name="electron")

    def deposits(self, E: int, n_steps: int) -> List[int]:
        E = int(max(1, E))
        n_steps = int(max(1, n_steps))

        # short effective range – tunable
        R = max(1, int(round(0.25 * E)) + 2)
        R = min(R, n_steps)

        i = np.arange(R, dtype=float)
        weights = np.exp(-1.3 * i)  # fast drop

        dose_R = _weights_to_int_dose(weights, E)
        out = dose_R + [0] * (n_steps - R)
        return out


class Gamma(Particle):
    """
    Gamma: buildup to a maximum at peak_index (arrow end), then attenuation.
    No explicit zero padding after the peak; deposits are defined across the full
    geometric path length n_steps.

    Shape is single-peaked by construction:
      - ramp (monotone increasing) from 0..peak_index
      - exponential attenuation (monotone decreasing) after peak_index
    """
    def __init__(self):
        super().__init__(name="gamma")

    def deposits(self, E: int, n_steps: int, peak_index: int | None = None) -> list[int]:
        E = int(max(1, E))
        n_steps = int(max(1, n_steps))

        # Choose peak: default to 35% depth if not provided (fallback)
        if peak_index is None:
            p = int(round(0.35 * (n_steps - 1)))
        else:
            p = int(np.clip(int(peak_index), 0, n_steps - 1))

        # Parameters (tunable):
        # - a: ramp exponent (higher => lower entrance, steeper buildup)
        # - tau: attenuation length after peak (higher => longer tail)
        a = 2.2
        tau = max(1.0, 0.35 * E)  # energy-dependent tail length

        w = np.zeros(n_steps, dtype=float)

        # --- buildup: strictly increasing to 1.0 at peak ---
        if p == 0:
            # peak at entrance: no buildup segment
            w[0] = 1.0
        else:
            x = np.arange(p + 1, dtype=float) / float(p)  # 0..1
            w[:p + 1] = x ** a
            w[p] = 1.0

        # --- attenuation: strictly decreasing after peak ---
        if p < n_steps - 1:
            k = np.arange(1, n_steps - p, dtype=float)  # 1..(n_steps-p-1)
            w[p + 1:] = np.exp(-k / tau)

        # Small floor avoids accidental zeros in weights (still may round to 0 in int dose)
        w += 1e-6

        doses = _weights_to_int_dose(w, E)  # length == n_steps, sum == E

        # --- enforce the maximum at p (avoid "two maxima" due to integer rounding) ---
        # If some bin exceeds doses[p], move units from the largest bins to the peak.
        if n_steps > 1:
            while True:
                max_other = max(doses[:p] + doses[p + 1:]) if (p > 0 or p < n_steps - 1) else 0
                if doses[p] >= max_other:
                    break
                # find an index with current maximum (excluding peak)
                j = None
                best = -1
                for idx, d in enumerate(doses):
                    if idx == p:
                        continue
                    if d > best:
                        best = d
                        j = idx
                if j is None or doses[j] <= 0:
                    break
                doses[j] -= 1
                doses[p] += 1

        return doses


