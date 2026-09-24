"""Manage WSDS block-cache eviction policies (rc.d style) + the systemd sweep.

The policy LIVES WITH THE DATASET: one JSON file at <dataset>/cache-policy.json
holding just the eviction parameters — everything location-like is fixed
relative to the dataset (the .sparse mirrors live next to the .wsds shards in each
partition's audio/ subfolder, the access logs in <dataset>/audio/.access):

    { "target_gb": 500,
      "consume_logs": true }

A dataset is REGISTERED with the eviction service by symlinking its policy
into the service dir (default ~/.config/wsds/evict.d/, override
WSDS_CACHE_POLICY_DIR):

    evict.d/data-en.json -> /mnt/weka/data-wsds/data-en/source/cache-policy.json

rc.d style: `disable` renames the symlink to <name>.json.disabled (the policy
file itself is untouched and shared by every user who linked it); `remove`
deletes only the symlink.

CLI (python -m wsds.cachectl ... , or `wsds-cache ...` when pip-installed):

    add       write <dataset>/cache-policy.json + register it   (--dataset, --target-gb, [--name])
    list      show registered policies + current size
    enable    | disable    | remove <name>
    run       evict all enabled policies (what the systemd service runs)  [--name X --dry-run]
    install-systemd   write a user timer+service and print the enable commands
    path      print the service (registration) dir
"""

import argparse
import glob
import json
import os
import re
import sys

POLICY_BASENAME = "cache-policy.json"
AUDIO_SUBDIR = "audio"  # fixed: mirrors in */audio/, logs in <dataset>/audio/.access


# ------------------------------------------------------------------ policy files
def policy_dir():
    d = os.environ.get("WSDS_CACHE_POLICY_DIR") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "wsds", "evict.d"
    )
    os.makedirs(d, exist_ok=True)
    return d


def _link_path(name, disabled=False):
    return os.path.join(policy_dir(), f"{name}.json" + (".disabled" if disabled else ""))


def _derive_name(dataset):
    p = dataset.rstrip("/")
    base = os.path.basename(p)
    if base in ("source", "blocks", ".wsds-cache", "audio"):  # climb to the informative dir
        base = os.path.basename(os.path.dirname(p)) or base
    return re.sub(r"[^A-Za-z0-9._-]", "-", base) or "cache"


def resolve_root(link_path):
    """Cache root for a registered policy: the dataset root, i.e. wherever the symlink's
    target really lives. Mirrors sit at <root>/<partition>/audio/<shard>.wsds.sparse and are
    named relative to the root (see WSS3Shard._resolve_cache); the logs are at <root>/audio/.access."""
    return os.path.dirname(os.path.realpath(link_path))


def resolve_logdir(root):
    return os.path.join(root, AUDIO_SUBDIR, ".access")


def iter_policies(include_disabled=True):
    """Yield (name, link_path, policy_dict, enabled) for registered policies."""
    out = []
    pats = [os.path.join(policy_dir(), "*.json")]
    if include_disabled:
        pats.append(os.path.join(policy_dir(), "*.json.disabled"))
    for pat in pats:
        for path in sorted(glob.glob(pat)):
            enabled = not path.endswith(".disabled")
            name = os.path.basename(path)[: -len(".json") if enabled else -len(".json.disabled")]
            try:
                with open(path) as f:
                    out.append((name, path, json.load(f), enabled))
            except Exception as e:
                print(f"  ! skipping unreadable policy {path}: {e}", file=sys.stderr)
    return out


# ------------------------------------------------------------------ subcommands
def cmd_add(a):
    dataset = os.path.abspath(a.dataset.rstrip("/"))
    if not os.path.isdir(dataset):
        sys.exit(f"add: no such dataset dir: {dataset}")
    name = a.name or _derive_name(dataset)
    link = _link_path(name)
    if (os.path.lexists(link) or os.path.lexists(_link_path(name, disabled=True))) and not a.force:
        sys.exit(f"add: {name} already registered (use --force to replace): {link}")
    pol_path = os.path.join(dataset, POLICY_BASENAME)
    pol = {"target_gb": a.target_gb, "consume_logs": not a.no_consume_logs}
    tmp = pol_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pol, f, indent=2)
        f.write("\n")
    os.replace(tmp, pol_path)
    for p in (link, _link_path(name, disabled=True)):
        if os.path.lexists(p):
            os.unlink(p)
    os.symlink(pol_path, _link_path(name, disabled=a.disabled))
    print(
        f"wrote {pol_path}\n  registered as {name} ({'disabled' if a.disabled else 'enabled'})\n"
        f"  cache root: {dataset} (mirrors under */{AUDIO_SUBDIR}/, logs in {AUDIO_SUBDIR}/.access)  target {a.target_gb} GB"
    )


def cmd_list(a):
    from .cache_evict import SLOT, scan_cache

    pols = iter_policies()
    if not pols:
        print(f"(no policies registered in {policy_dir()})")
        return
    print(f"{policy_dir()}\n")
    print(f"  {'NAME':20s} {'EN':3s} {'TARGET':>8s} {'PRESENT':>9s}  ROOT")
    for name, path, pol, enabled in pols:
        root = resolve_root(path)
        cur = "—"
        if a.sizes:
            try:
                cur = f"{sum(len(p) for _, _, p in scan_cache(root)) * SLOT / 1e9:.1f}G"
            except Exception:
                cur = "err"
        en = "on" if enabled else "off"
        print(f"  {name:20s} {en:3s} {str(pol.get('target_gb')) + 'G':>8s} {cur:>9s}  {root}")
    if not a.sizes:
        print("\n  (add --sizes to measure present bytes per cache — scans each tree)")


def _toggle(name, on):
    src, dst = _link_path(name, disabled=on), _link_path(name, disabled=not on)
    if os.path.lexists(dst):
        print(f"{name}: already {'enabled' if on else 'disabled'}")
        return
    if not os.path.lexists(src):
        sys.exit(f"no such policy: {name}")
    os.rename(src, dst)
    print(f"{name}: {'enabled' if on else 'disabled'}")


def cmd_enable(a):
    _toggle(a.name, True)


def cmd_disable(a):
    _toggle(a.name, False)


def cmd_remove(a):
    for p in (_link_path(a.name), _link_path(a.name, disabled=True)):
        if os.path.lexists(p):
            os.unlink(p)
            print(f"unregistered {a.name} ({p}; the dataset's {POLICY_BASENAME} is untouched)")
            return
    sys.exit(f"no such policy: {a.name}")


def cmd_run(a):
    from .cache_evict import evict_one

    pols = iter_policies()
    if a.name:
        pols = [p for p in pols if p[0] == a.name] or sys.exit(f"no such policy: {a.name}")
    ran = 0
    for name, path, pol, enabled in pols:
        if not a.name and not enabled:
            continue
        root = resolve_root(path)
        print(f"[{name}]")
        try:
            evict_one(
                root,
                float(pol["target_gb"]),
                logdir=resolve_logdir(root),
                dry_run=a.dry_run,
                consume_logs=bool(pol.get("consume_logs", True)),
            )
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
        ran += 1
    if not ran:
        print("no enabled policies to run.")


UNIT_SERVICE = """\
[Unit]
Description=WSDS block-cache LRU eviction sweep
After=network.target

[Service]
Type=oneshot
Environment=PYTHONPATH={pythonpath}
{policy_env}ExecStart={python} -m wsds.cachectl run
"""

UNIT_TIMER = """\
[Unit]
Description=Periodic WSDS block-cache eviction

[Timer]
OnCalendar={schedule}
Persistent=true

[Install]
WantedBy=timers.target
"""


def cmd_install_systemd(a):
    import wsds

    unit_dir = a.unit_dir or os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    pythonpath = os.path.dirname(os.path.dirname(os.path.abspath(wsds.__file__)))  # active checkout
    # carry a non-default policy dir into the service env so it sees the same policies
    pdir = os.environ.get("WSDS_CACHE_POLICY_DIR")
    policy_env = f"Environment=WSDS_CACHE_POLICY_DIR={pdir}\n" if pdir else ""
    svc = UNIT_SERVICE.format(pythonpath=pythonpath, python=sys.executable, policy_env=policy_env)
    tmr = UNIT_TIMER.format(schedule=a.schedule)
    sp = os.path.join(unit_dir, "wsds-cache-evict.service")
    tp = os.path.join(unit_dir, "wsds-cache-evict.timer")
    with open(sp, "w") as f:
        f.write(svc)
    with open(tp, "w") as f:
        f.write(tmr)
    print(f"wrote:\n  {sp}\n  {tp}\n")
    print("service:\n" + "\n".join("    " + line for line in svc.splitlines()))
    print(f"\nenable it (runs {a.schedule}):")
    print("    systemctl --user daemon-reload")
    print("    systemctl --user enable --now wsds-cache-evict.timer")
    print("    loginctl enable-linger $USER   # so the timer fires without an active login session")
    print("    systemctl --user list-timers wsds-cache-evict.timer   # verify")


def cmd_path(a):
    print(policy_dir())


def build_parser():
    ap = argparse.ArgumentParser(prog="wsds-cache", description="manage WSDS block-cache eviction policies")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="write <dataset>/cache-policy.json and register it with the service")
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset dir; the policy file lands in its root, mirrors are found under it recursively",
    )
    p.add_argument("--target-gb", type=float, required=True)
    p.add_argument("--name", help="registration name (default derived from the dataset path)")
    p.add_argument("--no-consume-logs", action="store_true", help="keep access logs after a pass")
    p.add_argument("--disabled", action="store_true", help="register it disabled")
    p.add_argument("--force", action="store_true", help="replace an existing registration")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("list", help="show policies")
    p.add_argument("--sizes", action="store_true", help="also measure present bytes per cache")
    p.set_defaults(func=cmd_list)

    for nm, fn, hlp in [
        ("enable", cmd_enable, "enable a policy"),
        ("disable", cmd_disable, "disable a policy"),
        ("remove", cmd_remove, "delete a policy"),
    ]:
        p = sub.add_parser(nm, help=hlp)
        p.add_argument("name")
        p.set_defaults(func=fn)

    p = sub.add_parser("run", help="evict enabled policies (systemd entry point)")
    p.add_argument("--name", help="run just this policy (even if disabled)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("install-systemd", help="write a user timer+service")
    p.add_argument(
        "--schedule", default="daily", help="systemd OnCalendar (default 'daily'; e.g. 'hourly', '*-*-* 04:00:00')"
    )
    p.add_argument("--unit-dir", default=None, help="where to write units (default ~/.config/systemd/user)")
    p.set_defaults(func=cmd_install_systemd)

    sub.add_parser("path", help="print the policy dir").set_defaults(func=cmd_path)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
