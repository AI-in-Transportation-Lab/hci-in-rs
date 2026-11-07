"""Chapter 1 - Worked Example 1.1: The Affordable Glance.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: code/glance_budget.py (shown in chapters/01a.tex).
Run from the repository root with `make ch01` or `python scripts/run_all.py --chapter 1`;
outputs are written to outputs/ch01/.

Worked Example 1.1 -- The affordable glance.

How long may an interface take the driver's eyes off the road
before the road itself can no longer absorb the error?
Closed-form kinematics only: no data, nothing to calibrate.
"""
A_BRAKE   = 3.4   # m/s^2, deceleration used for sight distance
T_REACT   = 1.0   # s, reaction time once the eyes are back
NHTSA_MAX = 2.0   # s, longest single glance in the guidelines


def affordable_glance(speed_kmh, sight_distance_m):
    """Longest off-road glance that still leaves room to stop."""
    v = speed_kmh / 3.6                 # m/s
    d_react = v * T_REACT               # road used while reacting
    d_brake = v ** 2 / (2 * A_BRAKE)    # road used while braking
    d_spare = sight_distance_m - d_react - d_brake
    return d_spare / v                  # negative: already too late


SITES = [("urban street",   30,  45),
         ("urban arterial", 50,  75),
         ("rural two-lane", 80, 150),
         ("freeway, crest", 110, 200)]

print(f"{'site':<16}{'km/h':>6}{'sight':>8}"
      f"{'blind @2s':>12}{'affordable':>13}{'verdict':>12}")

for name, kmh, sight in SITES:
    t_max = affordable_glance(kmh, sight)
    blind = kmh / 3.6 * NHTSA_MAX       # metres spent in a 2 s glance
    verdict = "ok" if t_max >= NHTSA_MAX else "OVER BUDGET"
    print(f"{name:<16}{kmh:>6}{sight:>7} m{blind:>10.1f} m"
          f"{t_max:>12.2f} s{verdict:>12}")
