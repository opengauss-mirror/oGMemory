# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Cost analysis and reporting for ContextEngine extractions.

This script analyzes trace files to generate cost reports by account and
memory type, with anomaly detection for unusual extraction patterns.

Usage:
    cd /Users/yp1017/projects/ContextEngine
    python scripts/cost_report.py [--trace-dir DIR] [--since TIMESTAMP]
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AccountCostSummary:
    """Cost summary for a single account."""

    account_id: str
    total_calls: int = 0
    total_latency: float = 0.0
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: int = 0
    anomaly: bool = False

    def avg_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency / self.total_calls if self.total_calls > 0 else 0.0


class CostReporter:
    """Generate cost reports from trace data."""

    def __init__(self, trace_dir: str = ".trace"):
        """Initialize the cost reporter.

        Args:
            trace_dir: Directory containing trace files.
        """
        self.trace_dir = Path(trace_dir)

    def generate_report(self, since_timestamp: int = 0) -> str:
        """Generate a markdown cost report.

        Args:
            since_timestamp: Unix timestamp to filter traces from.

        Returns:
            Markdown formatted report string.
        """
        # Aggregate trace data
        summaries = self._aggregate_traces(since_timestamp)

        if not summaries:
            return "# Extraction Cost Report\n\nNo trace data found.\n"

        # Detect anomalies
        self._detect_anomalies(summaries)

        # Generate report sections
        lines = [
            "# Extraction Cost Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Trace directory: {self.trace_dir}",
            f"Accounts analyzed: {len(summaries)}",
            "",
        ]

        # By Account section
        lines.extend([
            "## By Account",
            "",
            "| Account | Calls | Avg Latency | Errors | Anomaly |",
            "|---------|-------|-------------|--------|---------|",
        ])

        for account_id in sorted(summaries.keys()):
            summary = summaries[account_id]
            avg_ms = summary.avg_latency() * 1000
            anomaly_mark = "⚠️" if summary.anomaly else ""
            lines.append(
                f"| {account_id:20} | {summary.total_calls:6} | "
                f"{avg_ms:8.1f}ms | {summary.errors:6} | {anomaly_mark:7} |"
            )

        lines.extend(["", "## By Memory Type", "", "| Type | Calls | Total Latency |", "|------|-------|---------------|"])

        # Aggregate by type across all accounts
        type_totals: Dict[str, Dict[str, Any]] = {}
        for summary in summaries.values():
            for memory_type, stats in summary.by_type.items():
                if memory_type not in type_totals:
                    type_totals[memory_type] = {"calls": 0, "latency": 0.0}
                type_totals[memory_type]["calls"] += stats["calls"]
                type_totals[memory_type]["latency"] += stats["latency"]

        for memory_type in sorted(type_totals.keys()):
            totals = type_totals[memory_type]
            total_ms = totals["latency"] * 1000
            lines.append(f"| {memory_type:12} | {totals['calls']:6} | {total_ms:10.1f}ms |")

        lines.append("")

        # Anomalies section
        anomalies = [s for s in summaries.values() if s.anomaly]
        if anomalies:
            lines.extend([
                "## Anomalies",
                "",
                "The following accounts show extraction patterns >2x the average:",
                "",
            ])
            for summary in sorted(anomalies, key=lambda s: s.total_calls, reverse=True):
                lines.append(f"**{summary.account_id}**: {summary.total_calls} calls detected")
            lines.append("")

        # Summary statistics
        total_calls = sum(s.total_calls for s in summaries.values())
        total_errors = sum(s.errors for s in summaries.values())
        total_latency = sum(s.total_latency for s in summaries.values())
        global_avg = total_latency / total_calls if total_calls > 0 else 0

        lines.extend([
            "## Summary",
            "",
            f"- Total extraction calls: {total_calls}",
            f"- Total errors: {total_errors}",
            f"- Global average latency: {global_avg * 1000:.1f}ms",
            f"- Anomalous accounts: {len(anomalies)}",
            "",
        ])

        return "\n".join(lines)

    def _aggregate_traces(self, since: int) -> Dict[str, AccountCostSummary]:
        """Aggregate trace data by account.

        Args:
            since: Unix timestamp to filter from.

        Returns:
            Dict mapping account_id to AccountCostSummary.
        """
        summaries: Dict[str, AccountCostSummary] = {}

        if not self.trace_dir.exists():
            return summaries

        # Process trace files
        for trace_file in self.trace_dir.glob("*.json"):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)

                # Filter by timestamp
                trace_time = trace_data.get("timestamp", 0)
                if trace_time < since:
                    continue

                # Extract relevant fields
                account_id = trace_data.get("account_id", "unknown")
                if account_id not in summaries:
                    summaries[account_id] = AccountCostSummary(account_id=account_id)

                summary = summaries[account_id]

                # Update call count
                calls = trace_data.get("extraction_calls", 1)
                summary.total_calls += calls

                # Update latency
                latency = trace_data.get("total_latency_ms", 0) / 1000.0
                summary.total_latency += latency

                # Update by type
                memory_type = trace_data.get("memory_type", "unknown")
                if memory_type not in summary.by_type:
                    summary.by_type[memory_type] = {"calls": 0, "latency": 0.0}
                summary.by_type[memory_type]["calls"] += calls
                summary.by_type[memory_type]["latency"] += latency

                # Track errors
                if trace_data.get("error"):
                    summary.errors += 1

            except (json.JSONDecodeError, KeyError, IOError):
                # Skip malformed trace files
                continue

        return summaries

    def _detect_anomalies(self, summaries: Dict[str, AccountCostSummary]) -> None:
        """Detect accounts with anomalous extraction patterns.

        An account is flagged if it has >2x the average call count.

        Args:
            summaries: Dict of account summaries to analyze in-place.
        """
        if not summaries:
            return

        avg_calls = statistics.mean(s.total_calls for s in summaries.values())
        threshold = avg_calls * 2

        for summary in summaries.values():
            summary.anomaly = summary.total_calls > threshold


def main():
    """Main entry point for the cost report script."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate ContextEngine cost reports")
    parser.add_argument(
        "--trace-dir",
        default=".trace",
        help="Directory containing trace files (default: .trace)",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=0,
        help="Unix timestamp to filter traces from (default: 0, all time)",
    )

    args = parser.parse_args()

    # Create reporter and generate report
    reporter = CostReporter(trace_dir=args.trace_dir)
    report = reporter.generate_report(since_timestamp=args.since)

    print(report)
    print("=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
