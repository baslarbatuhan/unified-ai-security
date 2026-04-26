"""Monitoring — SLO checks, alert rule evaluation, metric snapshots.

Consumers (dashboard backend, cron jobs) call `alert_rules.evaluate(snapshot)`
to turn a plain metric dict into a list of fired Alert objects. The metric
snapshot itself can be built from telemetry with
`alert_rules.build_snapshot_from_events(events)`.
"""
