---
title: "12. Policy, Standards, and Implementation"
parent: Chapters
nav_order: 12
---

# Chapter 12: Policy, Standards, and Implementation
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

Each script below joins that case study's listings from the book in order, so it runs top to bottom. Run it with `python scripts/run_all.py --chapter 12` from the repository root; outputs go to `outputs/ch12/`.

## Case Study 1: Using HCI Evidence in Procurement for Safety-Critical Interfaces

Check three driver-monitoring bids against requirements using confidence bounds, compare their value under uncertainty, and design the contract's acceptance test.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs1_procurement.py](cs1_procurement.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch12_policy_implementation/cs1_procurement.py){: .btn .fs-3 }

![Using HCI Evidence in Procurement for Safety-Critical Interfaces]({{ '/docs/figures/chap12_fig01_procurement.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs1_procurement.py %}
```

## Case Study 2: Implementation Framework for Agency and Industry Deployment

Set Bayesian gates from pilot to full deployment, estimate population impact with RE-AIM, and detect a bad software update with CUSUM surveillance.

**Libraries:** NumPy, pandas, SciPy, matplotlib

[Download cs2_implementation.py](cs2_implementation.py){: .btn .btn-primary .fs-3 .mr-2 } [View on GitHub](https://github.com/AI-in-Transportation-Lab/hci-in-rs/blob/main/chapters/ch12_policy_implementation/cs2_implementation.py){: .btn .fs-3 }

![Implementation Framework for Agency and Industry Deployment]({{ '/docs/figures/chap12_fig02_implementation.png' | relative_url }})
*Figure as printed in the book.*

```python
{% include_relative cs2_implementation.py %}
```
