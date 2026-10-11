"""Replication of Vial, Shakkottai & Srikant, "Robust Multi-Agent Multi-Armed Bandits" (arXiv:2007.03812v3).

Engine modules (see REPLICATION_PLAN.md):
  ucb, engine, sticky   — Algorithm 1/2 core, UCB(alpha), sticky + active sets
  blocking, malicious, baselines — added later on top of `engine.run_gossip`
"""

__version__ = "0.0.1"
