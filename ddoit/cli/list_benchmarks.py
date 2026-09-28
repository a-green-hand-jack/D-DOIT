"""List registered D-DOIT benchmarks."""

from __future__ import annotations

from ddoit import BENCHMARKS


def main() -> None:
    for name, spec in BENCHMARKS.items():
        tasks = ", ".join(spec.supported_tasks)
        print(f"{name}: {spec.display_name} [{tasks}]")


if __name__ == "__main__":
    main()
