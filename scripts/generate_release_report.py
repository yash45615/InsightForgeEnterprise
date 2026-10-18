from __future__ import annotations

import sys

from insightforge.evaluation.release_report import generate_release_report


def main() -> int:
    try:
        output_path, report = generate_release_report()
    except (OSError, ValueError) as exc:
        print(f"Release report generation failed: {exc}", file=sys.stderr)
        return 2

    print(f"Report: {output_path}")
    print(f"Release status: {report['status']}")

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"[{status}] {check['metric']}: "
            f"{check['actual']:.3f} "
            f"(minimum {check['threshold']:.3f})"
        )

    if report["failed_checks"]:
        print("\nRelease gate failed:")
        for metric in report["failed_checks"]:
            print(f"- {metric}")

        return 1

    print("\nRelease gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
