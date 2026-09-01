from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import api, benchmark as benchmark_module, doctor as doctor_module, proxy
from .config import (
    ConfigurationError,
    DEFAULT_PROFILE,
    DEFAULT_STACK_PRESET,
    ROOT,
    list_runtime_profiles,
    load_model,
    load_profile,
    load_runtime,
)
from .install import install
from .lifecycle import gate as lifecycle_gate
from .models import adopt_model, download_model, verify_model
from .service import logs, start, status, stop, wait
from .service import _native_environment
from .verify import verify_python_environment, verify_runtime
from .backends import runtime_backend
from .manifest import recipe_source_root, recipe_venv


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _common_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="self-contained profile from profiles/production",
    )


def _tests(args: argparse.Namespace) -> None:
    profile = args.profile

    def unit() -> None:
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT,
            check=True,
        )
    def host() -> None:
        doctor_module.doctor(profile)

    def runtime() -> None:
        runtime_profile = load_runtime(profile)
        backend = runtime_backend(runtime_profile)
        if backend == "vllm":
            venv = recipe_venv(runtime_profile["recipe"])
            expected = venv / "bin" / "python"
        if backend == "vllm" and Path(sys.prefix).resolve() != venv.resolve():
            subprocess.run(
                [expected, str(ROOT / "run"), "test", "runtime", "--profile", profile],
                cwd=ROOT,
                check=True,
            )
            return
        if backend == "vllm":
            verify_python_environment(runtime_profile["recipe"])
        verify_runtime(profile, profile)

    def patch() -> None:
        runtime_profile = load_runtime(profile)
        python = recipe_venv(runtime_profile["recipe"]) / "bin" / "python"
        tree = recipe_source_root(runtime_profile["recipe"]) / "vllm"
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py",
                "-k",
                "rocm_dsv4_prefill_chunk_size or max_prefill_buffer_size or "
                "single_max_length_prefill or max_indexer_decode_workspace_rows",
                "-q",
            ],
            cwd=tree,
            check=True,
        )
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "tests/v1/spec_decode/test_dspark_prefill_suffix.py",
                "-q",
            ],
            cwd=tree,
            check=True,
        )

    def api_test() -> None:
        api.test_api(
            url=args.url,
            model_name=profile,
            runtime_name=profile,
            output=args.output,
            timeout=args.timeout,
        )

    def gpu() -> None:
        runtime_profile = load_runtime(profile)
        world = runtime_profile["parallel"]["tensor"] * runtime_profile["parallel"]["pipeline"]
        executable = recipe_venv(runtime_profile["recipe"]) / "bin" / "torchrun"
        environment = _native_environment(runtime_profile)
        for offset, mode in enumerate(("allreduce", "pipeline")):
            subprocess.run(
                [
                    executable,
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={world}",
                    f"--master-port={args.master_port + offset}",
                    ROOT / "r9700" / "gpu_worker.py",
                    "--mode",
                    mode,
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )

    def lifecycle() -> None:
        if args.output is None:
            raise ConfigurationError("lifecycle test requires --output")
        lifecycle_gate(
            model_name=profile,
            runtime_name=profile,
            ready_timeout=args.ready_timeout,
            stop_timeout=args.stop_timeout,
            output=args.output,
        )

    actions = {
        "unit": unit,
        "host": host,
        "runtime": runtime,
        "patch": patch,
        "gpu": gpu,
        "api": api_test,
        "lifecycle": lifecycle,
    }
    if args.tier == "all":
        for name in ("unit", "host", "runtime", "patch"):
            actions[name]()
    else:
        actions[args.tier]()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="r9700", description="Reproducible production R9700 runtime"
    )
    commands = root.add_subparsers(dest="command", required=True)

    install_parser = commands.add_parser("install", help="build the exact pinned runtime")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--jobs", type=int)
    install_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild from the local immutable recipe even when attested",
    )
    install_parser.add_argument(
        "--backend", choices=("all", "vllm", "llama-cpp"), default="all"
    )
    install_parser.add_argument(
        "--profile", help="install the recipe selected by a production profile"
    )
    install_parser.add_argument(
        "--recipe", help="install one named immutable runtime recipe"
    )

    doctor_parser = commands.add_parser("doctor", help="validate host prerequisites")
    doctor_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    doctor_parser.add_argument("--output", type=_path)

    config_parser = commands.add_parser("config", help="resolve a profile")
    config_parser.add_argument("kind", choices=("profile", "model", "runtime"))
    config_parser.add_argument("name")

    profiles_parser = commands.add_parser(
        "profiles", help="list or inspect lifecycle-classified runtime profiles"
    )
    profiles_commands = profiles_parser.add_subparsers(
        dest="profiles_command", required=True
    )
    profiles_list = profiles_commands.add_parser("list")
    profiles_list.add_argument("--json", action="store_true")
    profiles_show = profiles_commands.add_parser("show")
    profiles_show.add_argument("name")

    model_parser = commands.add_parser("model", help="manage checkpoint data")
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    for action in ("adopt", "download", "verify"):
        item = model_commands.add_parser(action)
        item.add_argument("name", nargs="?", default=DEFAULT_PROFILE)
        item.add_argument("--directory")
        if action == "download":
            item.add_argument("--dry-run", action="store_true")

    service_parser = commands.add_parser("service", help="manage one identity-bound server")
    service_commands = service_parser.add_subparsers(dest="service_command", required=True)
    service_start = service_commands.add_parser("start")
    _common_profile(service_start)
    service_start.add_argument("--model-directory")
    service_start.add_argument("--host")
    service_start.add_argument("--port", type=int)
    service_start.add_argument("--backend", choices=("vllm", "llama-cpp"))
    service_start.add_argument("--wait", action="store_true")
    service_start.add_argument("--ready-timeout", type=float, default=900)
    service_wait = service_commands.add_parser("wait")
    service_wait.add_argument("--timeout", type=float, default=900)
    service_stop = service_commands.add_parser("stop")
    service_stop.add_argument("--timeout", type=float, default=180)
    service_logs = service_commands.add_parser("logs")
    service_logs.add_argument("--follow", action="store_true")
    service_logs.add_argument("--lines", type=int, default=100)
    service_commands.add_parser("status")

    proxy_parser = commands.add_parser("proxy", help="manage the LiteLLM proxy service")
    proxy_commands = proxy_parser.add_subparsers(dest="proxy_command", required=True)
    proxy_commands.add_parser("install")
    proxy_start = proxy_commands.add_parser("start")
    proxy_start.add_argument("--host")
    proxy_start.add_argument("--port", type=int)
    proxy_start.add_argument("--backend-url")
    proxy_start.add_argument("--wait", action="store_true")
    proxy_start.add_argument("--ready-timeout", type=float, default=120)
    proxy_wait = proxy_commands.add_parser("wait")
    proxy_wait.add_argument("--timeout", type=float, default=120)
    proxy_stop = proxy_commands.add_parser("stop")
    proxy_stop.add_argument("--timeout", type=float, default=30)
    proxy_logs = proxy_commands.add_parser("logs")
    proxy_logs.add_argument("--follow", action="store_true")
    proxy_logs.add_argument("--lines", type=int, default=100)
    proxy_commands.add_parser("status")
    proxy_test = proxy_commands.add_parser("test")
    proxy_test.add_argument("--timeout", type=float, default=120)

    stack_parser = commands.add_parser(
        "stack", help="manage a preset coding stack"
    )
    stack_commands = stack_parser.add_subparsers(
        dest="stack_command", required=True
    )
    stack_start = stack_commands.add_parser("start")
    stack_start.add_argument("--preset", default=DEFAULT_STACK_PRESET)
    stack_start.add_argument("--dry-run", action="store_true")
    for action in ("stop", "status", "presets"):
        stack_commands.add_parser(action)

    test_parser = commands.add_parser("test", help="run a named verification tier")
    test_parser.add_argument(
        "tier", choices=("unit", "host", "runtime", "patch", "gpu", "api", "lifecycle", "all")
    )
    _common_profile(test_parser)
    test_parser.add_argument("--url", default="http://127.0.0.1:8000")
    test_parser.add_argument("--timeout", type=float, default=600)
    test_parser.add_argument("--output", type=_path)
    test_parser.add_argument("--master-port", type=int, default=29571)
    test_parser.add_argument("--ready-timeout", type=float, default=900)
    test_parser.add_argument("--stop-timeout", type=float, default=180)

    bench = commands.add_parser("benchmark", help="measure fixed-size API workloads")
    bench.add_argument("--url", default="http://127.0.0.1:8000")
    _common_profile(bench)
    bench.add_argument("--prompt-tokens", type=int, required=True)
    bench.add_argument("--output-tokens", type=int, required=True)
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--repetitions", type=int, default=5)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--prompt-variant-offset", type=int, default=0)
    bench.add_argument("--pin-data-parallel", action="store_true")
    bench.add_argument("--data-parallel-rank", type=int)
    bench.add_argument("--timeout", type=float, default=900)
    bench.add_argument("--output", type=_path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "install":
            install(
                dry_run=args.dry_run,
                jobs=args.jobs,
                backend=args.backend,
                runtime_name=args.profile,
                recipe_name=args.recipe,
                rebuild=args.rebuild,
            )
        elif args.command == "doctor":
            doctor_module.doctor(args.profile, output=args.output)
        elif args.command == "config":
            value = {
                "profile": load_profile,
                "model": load_model,
                "runtime": load_runtime,
            }[args.kind](args.name)
            print(json.dumps(value, indent=2))
        elif args.command == "profiles":
            if args.profiles_command == "show":
                print(json.dumps(load_profile(args.name), indent=2))
            else:
                profiles = list_runtime_profiles()
                if args.json:
                    print(json.dumps(profiles, indent=2))
                else:
                    for profile in profiles:
                        print(
                            f"{profile['tier']:<10} {profile['status']:<24} "
                            f"{profile['name']}"
                        )
        elif args.command == "model":
            if args.model_command == "download":
                download_model(args.name, directory=args.directory, dry_run=args.dry_run)
            elif args.model_command == "adopt":
                adopt_model(args.name, directory=args.directory)
            else:
                print(json.dumps(verify_model(args.name, args.directory), indent=2))
        elif args.command == "service":
            if args.service_command == "start":
                start(
                    args.profile,
                    args.profile,
                    model_directory=args.model_directory,
                    host=args.host,
                    port=args.port,
                    backend=args.backend,
                    wait_ready=args.wait,
                    ready_timeout=args.ready_timeout,
                )
            elif args.service_command == "wait":
                wait(timeout=args.timeout)
            elif args.service_command == "stop":
                stop(timeout=args.timeout)
            elif args.service_command == "status":
                return status()
            else:
                logs(follow=args.follow, lines=args.lines)
        elif args.command == "proxy":
            if args.proxy_command == "install":
                proxy.install()
            elif args.proxy_command == "start":
                proxy.start(
                    host=args.host,
                    port=args.port,
                    backend_url=args.backend_url,
                    wait_ready=args.wait,
                    ready_timeout=args.ready_timeout,
                )
            elif args.proxy_command == "wait":
                proxy.wait(timeout=args.timeout)
            elif args.proxy_command == "stop":
                proxy.stop(timeout=args.timeout)
            elif args.proxy_command == "status":
                return proxy.status()
            elif args.proxy_command == "test":
                proxy.test(timeout=args.timeout)
            else:
                proxy.logs(follow=args.follow, lines=args.lines)
        elif args.command == "test":
            _tests(args)
        elif args.command == "stack":
            command = [
                str(
                    ROOT
                    / "skills"
                    / "manage-r9700-stack"
                    / "scripts"
                    / "manage-stack.sh"
                ),
                args.stack_command,
            ]
            if args.stack_command == "start":
                command.extend(("--preset", args.preset))
                if args.dry_run:
                    command.append("--dry-run")
            return subprocess.run(command, cwd=ROOT, check=False).returncode
        elif args.command == "benchmark":
            benchmark_module.benchmark(
                url=args.url,
                model_name=args.profile,
                runtime_name=args.profile,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                concurrency=args.concurrency,
                repetitions=args.repetitions,
                warmup=args.warmup,
                timeout=args.timeout,
                output=args.output,
                pin_data_parallel=args.pin_data_parallel,
                data_parallel_rank=args.data_parallel_rank,
                prompt_variant_offset=args.prompt_variant_offset,
            )
        return 0
    except (ConfigurationError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
