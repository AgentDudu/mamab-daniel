"""Replication of Vial, Shakkottai & Srikant, "Robust Multi-Agent Multi-Armed Bandits" (arXiv:2007.03812v3).

Engine modules (see REPLICATION_PLAN.md):
  rng, ucb, sticky       — splitmix64 streams, UCB(alpha)/phase rules, sticky + active sets
  engine                 — Algorithm 1 kernel (honest agents + Get-Rec)
  blocking, malicious    — Algorithm 3 blocklists, Algorithm 2 malicious strategies
  baselines              — no_comm (C15) + the four-algorithm dispatcher
"""

__version__ = "0.0.1"
