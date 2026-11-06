"""Command line: aneopt audit|plan|profile|probe."""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aneopt", description="Find Apple Neural Engine bottlenecks in CoreML models.")
    sub = ap.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="list activations over the on-chip limit; reads the package, needs no hardware")
    audit.add_argument("package")
    audit.add_argument("--limit-kib", type=int, help="on-chip limit, by default the largest size known safe on an M2 Pro")

    plan = sub.add_parser("plan", help="CoreML's per-op device placement and estimated cost")
    plan.add_argument("package")
    plan.add_argument("--unit", default="CPU_AND_NE")

    profile = sub.add_parser("profile", help="latency and Neural Engine memory traffic, with random inputs")
    profile.add_argument("package")
    profile.add_argument("--unit", default="CPU_AND_NE")
    profile.add_argument("--seconds", type=float, default=12)

    probe = sub.add_parser("probe", help="synthetic tests of this machine's Neural Engine")
    probe.add_argument("mode", choices=["sweep", "mlp", "grid"])
    probe.add_argument("--seconds", type=float, default=6)

    args = ap.parse_args(argv)
    if args.command == "audit":
        from .audit import DEFAULT_LIMIT_BYTES, print_audit

        print_audit(args.package, args.limit_kib * 1024 if args.limit_kib else DEFAULT_LIMIT_BYTES)
    elif args.command == "plan":
        from .plan import print_plan

        print_plan(args.package, args.unit)
    elif args.command == "profile":
        from .measure import profile_package

        print(json.dumps(profile_package(args.package, args.unit, args.seconds)))
    elif args.command == "probe":
        from .probe import find_step, run

        rows = run(args.mode, args.seconds)
        if args.mode == "sweep":
            print("working-set step:", find_step(rows))


if __name__ == "__main__":
    main()
