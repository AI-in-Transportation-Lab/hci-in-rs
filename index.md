---
title: Home
layout: home
nav_order: 1
permalink: /
---

# HCI in Roadway Safety
{: .fs-9 }

Runnable code for every case study in the book *Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications* by Subasish Das.
{: .fs-6 .fw-300 }

[Browse the chapters]({{ '/chapters/' | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs){: .btn .fs-5 .mb-4 .mb-md-0 }

---

<p align="center">
  <img src="{{ '/docs/figures/chap04_fig01_glance_usability.png' | relative_url }}" width="32%">
  <img src="{{ '/docs/figures/chap05_fig02_takeover.png' | relative_url }}" width="32%">
  <img src="{{ '/docs/figures/chap10_fig02_nds_claims.png' | relative_url }}" width="32%">
</p>

Every code listing in the book's case studies is here as a runnable Python script, organised by chapter, with the code shown on each chapter page. The scripts generate their own synthetic data, so you can reproduce each figure and printed result without data access agreements.

## Chapters

| Ch. | Title | Code |
|---|---|---|
| 1 | [Human Factors and Roadway Safety: Context and Challenges]({{ '/chapters/ch01_human_factors_context/' | relative_url }}) | The affordable glance |
| 2 | [Cognitive Foundations for Safe Driving Interaction]({{ '/chapters/ch02_cognitive_foundations/' | relative_url }}) | Driver workload with NASA-TLX and eye tracking; Trust calibration in decision support |
| 3 | [Distraction and Secondary Task Interaction]({{ '/chapters/ch03_distraction/' | relative_url }}) | Infotainment distraction in urban driving; Warning overload in safety-critical alerts |
| 4 | [In-Vehicle Interface Design for Safety-Critical Use]({{ '/chapters/ch04_in_vehicle_interfaces/' | relative_url }}) | Glance management and usability testing; Touch, voice, gesture and multimodal interfaces |
| 5 | [Driver Assistance and Partial Automation (SAE Levels 1-3)]({{ '/chapters/ch05_driver_assistance/' | relative_url }}) | Driver monitoring in Level 2 automation; Takeover request timing for Level 3 |
| 6 | [Full Autonomy and Human-Machine Interaction (SAE Levels 4-5)]({{ '/chapters/ch06_full_autonomy/' | relative_url }}) | eHMI intent signaling for pedestrians; Safety assurance for Level 4-5 deployment |
| 7 | [Human-AI Interaction in Safety-Critical Transportation Systems]({{ '/chapters/ch07_human_ai_interaction/' | relative_url }}) | Explainable AI dashboards; Bias audit of a crash-risk model |
| 8 | [Vulnerable Road Users and Inclusive Interface Design]({{ '/chapters/ch08_vulnerable_road_users/' | relative_url }}) | V2P safety alerts; Inclusive interface design |
| 9 | [Roadside Interfaces, Connected Infrastructure, and ITS]({{ '/chapters/ch09_roadside_its/' | relative_url }}) | Dynamic message signs in work zones; AI-assisted TMC alert triage |
| 10 | [Evaluation Methods for Safety-Critical HCI]({{ '/chapters/ch10_evaluation_methods/' | relative_url }}) | Simulator and field evaluation; Naturalistic driving evidence to safety claims |
| 11 | [Emerging Directions in HCI for Roadway Safety]({{ '/chapters/ch11_emerging_directions/' | relative_url }}) | Adaptive AI copilots; AR/VR training environments |
| 12 | [Policy, Standards, and Implementation]({{ '/chapters/ch12_policy_implementation/' | relative_url }}) | HCI evidence in procurement; Implementation framework for deployment |

## A taste of the code

The affordable glance (Chapter 1, Worked Example 1.1):

```python
def affordable_glance(speed_kmh, sight_distance_m):
    """Longest off-road glance that still leaves room to stop."""
    v = speed_kmh / 3.6                 # m/s
    d_react = v * T_REACT               # road used while reacting
    d_brake = v ** 2 / (2 * A_BRAKE)    # road used while braking
    d_spare = sight_distance_m - d_react - d_brake
    return d_spare / v                  # negative: already too late
```

## Quick start

See [Getting started]({{ '/getting-started/' | relative_url }}) for details.

```bash
git clone https://github.com/AI-in-Transportation-Lab/hci-in-rs.git
cd hci-in-rs
uv sync
uv run python scripts/run_all.py
```

Developed with the Artificial Intelligence in Transportation (AIT) Lab at Texas State University.
