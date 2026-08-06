"""Phase 1 verification script — CLI checks without the HTTP server.

Usage:
    python scripts/verify.py            # full Phase 1 verification
    python scripts/verify.py --hardware
    python scripts/verify.py --registry
    python scripts/verify.py --providers
"""

from __future__ import annotations

import argparse
import sys

from synapse.bootstrap import Boot, create_container


def check_hardware(boot: Boot) -> None:
    p = boot.hardware.scan()
    print("== Hardware ==")
    print(f"  OS:          {p.os_name} {p.os_version} ({p.arch})")
    print(f"  CPU:         {p.cpu.model} ({p.cpu.cores}c/{p.cpu.threads}t)")
    print(f"  RAM:         {p.memory.total_gb:.1f} GB total, {p.memory.available_gb:.1f} GB available")
    gpu = f"{p.gpu.name} ({p.gpu.vram_gb} GB)" if p.gpu else "not detected"
    print(f"  GPU:         {gpu}")
    print(f"  Storage:     {p.storage.free_gb:.1f} GB free")
    rec = p.recommendations
    print(f"  Local LLM:   {'yes' if rec.can_run_local_llm else 'no'} (max ~{rec.max_quantized_params_billions:.1f}B params)")


def check_registry(boot: Boot) -> None:
    models = boot.registry.all()
    print(f"== Model Registry ({len(models)} models) ==")
    for m in sorted(models, key=lambda m: m.provider_id):
        print(
            f"  {m.id:<24} provider={m.provider_id:<8} kind={m.kind.value:<6} "
            f"ctx={m.context_window or '?'}"
        )


def check_providers(boot: Boot) -> None:
    print("== Providers ==")
    if not boot.providers.provider_ids():
        print("  none enabled in config")
        return
    for pid in boot.providers.provider_ids():
        provider = boot.providers.get(pid)
        state = boot.providers.state(pid).value
        healthy = boot.providers.health(pid)
        print(f"  {pid:<10} kind={provider.kind.value:<6} state={state:<10} health={'OK' if healthy else 'DOWN'}")
        if not healthy:
            print("           -> check: is the service running? is base_url correct?")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synapse One Phase 1 verification")
    parser.add_argument("--hardware", action="store_true", help="only hardware check")
    parser.add_argument("--registry", action="store_true", help="only registry check")
    parser.add_argument("--providers", action="store_true", help="only providers check")
    args = parser.parse_args()

    boot = Boot(create_container())
    boot.start()
    try:
        if args.hardware:
            check_hardware(boot)
        elif args.registry:
            check_registry(boot)
        elif args.providers:
            check_providers(boot)
        else:
            check_hardware(boot)
            check_registry(boot)
            check_providers(boot)
    finally:
        boot.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
