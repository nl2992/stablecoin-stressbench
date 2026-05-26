"""Experiment definitions for Stablecoin StressBench.

An experiment is a (task, feature_set, model) triple. Tasks define what is
predicted (label + horizon + notional). Feature sets define which columns from
``dataset.parquet`` are used as inputs. Models are the estimators being
compared.

See ``tasks.py``, ``feature_sets.py``, and ``experiment_runner.py``.
"""
