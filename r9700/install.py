from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ROOT, ConfigurationError, load_runtime
from .manifest import (
    default_recipe_name,
    recipe_constraints_path,
    recipe_install_path,
    recipe_record,
    recipe_source_root,
    recipe_venv,
    runtime_manifest,
    tracked_diff_sha256,
    verify_assets,
    verify_install,
    write_install_manifest,
)

EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def _pip_command(python: Path) -> list[str | Path]:
    """Return a pip command that remains valid for relocated/minimal venvs."""
    return [python, "-m", "pip"]


def _make_venv_entrypoints_relocatable(venv: Path) -> None:
    """Replace absolute venv Python shebangs with a local-path trampoline."""
    launcher = (
        b"#!/bin/sh\n"
        b"'''exec' \"$(dirname -- \"$(realpath -- \"$0\")\")\"/'python' \"$0\" \"$@\"\n"
        b"' '''\n"
    )
    bin_directory = venv / "bin"
    if not bin_directory.is_dir():
        return
    for entrypoint in bin_directory.iterdir():
        if entrypoint.is_symlink() or not entrypoint.is_file():
            continue
        with entrypoint.open("rb") as stream:
            first_line = stream.readline()
            if not first_line.startswith(b"#!"):
                continue
            remainder = stream.read()
        if not first_line.endswith(b"\n"):
            continue
        try:
            interpreter = Path(first_line[2:].decode().strip().split(maxsplit=1)[0])
        except (UnicodeDecodeError, IndexError):
            continue
        if (
            not interpreter.is_absolute()
            or not interpreter.name.startswith("python")
            or interpreter.parent.name != "bin"
            or interpreter.parent.parent.name != "venv"
        ):
            continue
        entrypoint.write_bytes(launcher + remainder)


def _torch_install_requirements(environment: dict[str, Any], arch: str) -> list[str]:
    """Build the Torch/ROCm requirement set for SDK-wheel or system ROCm."""
    torch = environment["torch_version"]
    torchvision = environment["torchvision_version"]
    requirements = [f"torch=={torch}"]
    if torchaudio := environment.get("torchaudio_version"):
        requirements.append(f"torchaudio=={torchaudio}")
    requirements.append(f"torchvision=={torchvision}")
    if environment.get("rocm_root"):
        return requirements
    sdk_requirements = [
        f"rocm[libraries,devel,device-{arch}]=={environment['rocm_version']}",
        f"torch=={torch}",
        f"amd-torch-device-{arch}=={torch}",
    ]
    if torchaudio := environment.get("torchaudio_version"):
        sdk_requirements.append(f"torchaudio=={torchaudio}")
    sdk_requirements.extend(
        [
            f"torchvision=={torchvision}",
            f"amd-torchvision-device-{arch}=={torchvision}",
        ]
    )
    return sdk_requirements


def _amdsmi_install_requirement(environment: dict[str, Any]) -> str:
    """Return the pinned AMD SMI requirement, optionally from an exact wheel."""
    if url := environment.get("amdsmi_url"):
        digest = environment.get("amdsmi_sha256")
        if not digest:
            raise ConfigurationError("amdsmi_url requires amdsmi_sha256")
        return f"amdsmi @ {url}#sha256={digest}"
    return f"amdsmi=={environment['amdsmi_version']}"


def _activate_and_verify_source_package(
    *,
    python: Path,
    source: Path,
    distribution: str,
    import_name: str,
    env: dict[str, str],
) -> None:
    """Make a pinned checkout authoritative over dependency-installed wheels."""
    # vLLM's ROCm requirements may install an ``amd-aiter`` wheel after the
    # source checkout was prepared. Leaving both installed makes Python import
    # the wheel directory before setuptools' editable-source path, while the
    # install attestation still records the source commit. Remove that ambiguity
    # and install the exact checkout only after every dependency transaction.
    # Use ``python -m pip`` because a relocated venv can retain a stale shebang
    # in ``bin/pip`` even though its interpreter remains usable.
    run([*_pip_command(python), "uninstall", "--yes", distribution], env=env)
    run([python, "setup.py", "develop", "--no-deps"], cwd=source, env=env)

    import_probe = (
        "import importlib; from pathlib import Path; "
        f"module = importlib.import_module({import_name!r}); "
        "print(Path(module.__file__).resolve())"
    )
    result = subprocess.run(
        [
            str(python),
            "-c",
            import_probe,
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise ConfigurationError(f"{distribution} import probe returned no path")
    imported = Path(output_lines[-1]).resolve()
    expected_root = source.resolve()
    if not imported.is_relative_to(expected_root):
        raise ConfigurationError(
            f"{distribution} import does not resolve to the pinned source checkout: "
            f"{imported} is outside {expected_root}"
        )


def _activate_and_verify_aiter_source(
    *,
    python: Path,
    source: Path,
    env: dict[str, str],
) -> None:
    _activate_and_verify_source_package(
        python=python,
        source=source,
        distribution="amd-aiter",
        import_name="aiter",
        env=env,
    )


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=tree, check=True, capture_output=True, text=True
    ).stdout.strip()


def _apply_patch_image(
    tree: Path, patches: list[dict[str, Any]], *, start_index: int = 0
) -> None:
    for patch in patches[start_index:]:
        patch_path = ROOT / patch["path"]
        run(["git", "apply", "--check", patch_path], cwd=tree)
        run(["git", "apply", patch_path], cwd=tree)


def _local_aiter_submodule_reference(tree: Path, expected_commit: str) -> Path | None:
    """Return an exact, clean CK checkout from another installed recipe."""
    recipe_root = ROOT / ".runtime" / "recipes"
    for candidate in sorted(recipe_root.glob("*/src/aiter/3rdparty/composable_kernel")):
        if candidate.resolve() == (tree / "3rdparty" / "composable_kernel").resolve():
            continue
        try:
            if _git(candidate, "rev-parse", "HEAD") != expected_commit:
                continue
            if _git(candidate, "status", "--porcelain"):
                continue
        except (OSError, subprocess.CalledProcessError):
            continue
        return candidate
    return None


def _materialize_aiter_submodules(tree: Path) -> None:
    update_command = [
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
    ]
    expected_commit = _git(tree, "rev-parse", "HEAD:3rdparty/composable_kernel")
    target = tree / "3rdparty" / "composable_kernel"
    if (target / ".git").exists():
        try:
            if _git(target, "rev-parse", "HEAD") == expected_commit and not _git(
                target, "status", "--porcelain"
            ):
                print(
                    "aiter Composable Kernel already materialized at the exact "
                    "clean commit",
                    flush=True,
                )
                return
        except (OSError, subprocess.CalledProcessError):
            pass
        run(update_command, cwd=tree)
        return

    reference = _local_aiter_submodule_reference(tree, expected_commit)
    if reference is None:
        run(update_command, cwd=tree)
        return

    print(
        f"aiter Composable Kernel clones exact local objects from {reference}",
        flush=True,
    )
    run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(reference),
            str(target),
        ]
    )
    run(["git", "checkout", "--detach", expected_commit], cwd=target)
    run(
        ["git", "submodule", "absorbgitdirs", "3rdparty/composable_kernel"],
        cwd=tree,
    )


def installation_plan(recipe_name: str | None = None) -> list[str]:
    selected = recipe_name or default_recipe_name(backend="vllm")
    manifest = runtime_manifest(selected)
    rows = [
        f"recipe {selected}",
        f"Python {manifest['platform']['python']}",
        f"ROCm {manifest['environment']['rocm_version']} ({manifest['platform']['gpu_arch']})",
        f"PyTorch {manifest['environment']['torch_version']}",
    ]
    for name, source in manifest["sources"].items():
        rows.append(
            f"{name} {source['commit']} with {len(source['patches'])} ordered patches"
        )
    return rows


def install(
    *,
    dry_run: bool = False,
    jobs: int | None = None,
    backend: str = "all",
    runtime_name: str | None = None,
    recipe_name: str | None = None,
    rebuild: bool = False,
) -> None:
    if backend not in ("all", "vllm", "llama-cpp"):
        raise ConfigurationError(f"unsupported install backend: {backend}")
    if runtime_name and recipe_name:
        raise ConfigurationError("choose either --profile or --recipe")
    if runtime_name:
        selected = [str(load_runtime(runtime_name)["recipe"])]
    elif recipe_name:
        recipe_record(recipe_name)
        selected = [recipe_name]
    elif backend == "all":
        selected = [
            default_recipe_name(backend="vllm"),
            default_recipe_name(backend="llama-cpp"),
        ]
    else:
        selected = [default_recipe_name(backend=backend)]
    for recipe in selected:
        selected_backend = recipe_record(recipe)["backend"]
        if backend != "all" and selected_backend != backend:
            raise ConfigurationError(
                f"recipe {recipe} selects {selected_backend}, not {backend}"
            )
        if selected_backend == "vllm":
            _install_vllm(recipe, dry_run=dry_run, jobs=jobs, rebuild=rebuild)
        elif selected_backend == "llama-cpp":
            from .backends.llama_cpp import install as install_llama_cpp

            foundation = recipe_record(recipe).get("foundation_recipe")
            if foundation:
                _install_vllm(
                    str(foundation),
                    dry_run=dry_run,
                    jobs=jobs,
                    rebuild=False,
                )
            install_llama_cpp(
                dry_run=dry_run,
                jobs=jobs,
                recipe_name=recipe,
                rebuild=rebuild,
            )


def _install_vllm(
    recipe_name: str,
    *,
    dry_run: bool = False,
    jobs: int | None = None,
    rebuild: bool = False,
) -> None:
    manifest = runtime_manifest(recipe_name)
    verify_assets(manifest, recipe_name=recipe_name)
    for row in installation_plan(recipe_name):
        print(f"INSTALL {row}")
    if dry_run:
        for name, source in manifest["sources"].items():
            for index, patch in enumerate(source["patches"], 1):
                print(f"  {name}[{index:02d}] {patch['path']} {patch['sha256']}")
        return
    if not rebuild:
        try:
            installed = verify_install(recipe_name)
            venv = recipe_venv(recipe_name)
            if not (venv / "bin" / "vllm").is_file():
                raise ConfigurationError("attested vLLM executable is absent")
            _make_venv_entrypoints_relocatable(venv)
            print(
                f"runtime recipe already installed: {recipe_name} "
                f"({installed['runtime_manifest_sha256']})"
            )
            return
        except ConfigurationError:
            pass

    expected_python = manifest["platform"]["python"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise ConfigurationError(
            f"installer requires Python {expected_python}, found {actual_python}"
        )
    for executable in ("uv", "git", "gcc", "g++", "cmake", "ninja"):
        if shutil.which(executable) is None:
            raise ConfigurationError(
                f"required build tool is absent: {executable}; "
                "run scripts/bootstrap-host.sh"
            )

    environment = manifest["environment"]
    venv = recipe_venv(recipe_name)
    sources = recipe_source_root(recipe_name)
    constraints = recipe_constraints_path(recipe_name)
    build_jobs = jobs or int(os.environ.get("NATIVE_BUILD_JOBS", os.cpu_count() or 1))

    run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            "--allow-existing",
            "--relocatable",
            venv,
        ]
    )
    python = venv / "bin" / "python"
    # Venvs created from relocated or externally managed interpreters may not
    # provide a usable ``bin/pip`` script. The module entrypoint is tied to the
    # selected interpreter and works without relying on a generated shebang.
    pip = _pip_command(python)
    constraint_args = ["--constraint", constraints]
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *constraint_args,
            f"pip=={environment['pip_version']}",
            f"setuptools=={environment['setuptools_version']}",
            f"wheel=={environment['wheel_version']}",
        ]
    )
    arch = manifest["platform"]["gpu_arch"]
    run(
        [
            *pip,
            "install",
            *constraint_args,
            "--index-url",
            environment["rocm_index_url"],
            "--extra-index-url",
            "https://pypi.org/simple",
            *_torch_install_requirements(environment, arch),
        ]
    )
    if configured_rocm_root := environment.get("rocm_root"):
        rocm_root = Path(configured_rocm_root).resolve()
        if not rocm_root.is_dir():
            raise ConfigurationError(f"system ROCm root is absent: {rocm_root}")
        rocm_home = str(rocm_root)
    else:
        run([venv / "bin" / "rocm-sdk", "init"])
        rocm_home = subprocess.run(
            [venv / "bin" / "rocm-sdk", "path", "--root"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    build_env = os.environ.copy()
    build_env.update(
        {
            "ROCM_HOME": rocm_home,
            "ROCM_PATH": rocm_home,
            "PATH": f"{rocm_home}/bin:{venv / 'bin'}:{build_env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{rocm_home}/lib:{build_env.get('LD_LIBRARY_PATH', '')}",
            "PYTORCH_ROCM_ARCH": arch,
            "GPU_ARCHS": arch,
            "TRITON_DEFAULT_BACKEND": "amd",
            "MAX_JOBS": str(build_jobs),
        }
    )
    sources.mkdir(parents=True, exist_ok=True)

    aiter = _prepare_source("aiter", sources / "aiter", manifest)
    run(
        [
            *pip,
            "install",
            *constraint_args,
            "--requirement",
            aiter / "requirements.txt",
        ],
        env=build_env,
    )
    aiter_env = dict(build_env, AITER_USE_SYSTEM_TRITON="1", PREBUILD_KERNELS="0")

    if "triton-kernels" in manifest["sources"]:
        triton_source = _prepare_source(
            "triton-kernels", sources / "triton-kernels", manifest
        )
        triton_kernels = triton_source / "python" / "triton_kernels" / "triton_kernels"
        if not triton_kernels.is_dir():
            raise ConfigurationError(
                f"Triton kernels source layout is unexpected: {triton_kernels}"
            )
        build_env["TRITON_KERNELS_SRC_DIR"] = str(triton_kernels)

    vllm = _prepare_source("vllm", sources / "vllm", manifest)
    run(
        [
            *pip,
            "install",
            *constraint_args,
            "--requirement",
            vllm / "requirements" / "rocm.txt",
        ],
        env=build_env,
    )
    run(
        [
            *pip,
            "install",
            *constraint_args,
            _amdsmi_install_requirement(environment),
            f"huggingface_hub[hf_xet]=={environment['huggingface_hub_version']}",
            "tblib==3.1.0",
        ],
        env=build_env,
    )
    _activate_and_verify_aiter_source(
        python=python,
        source=aiter,
        env=aiter_env,
    )
    _activate_and_verify_source_package(
        python=python,
        source=vllm,
        distribution="vllm",
        import_name="vllm",
        env=dict(build_env, VLLM_TARGET_DEVICE="rocm"),
    )
    _make_venv_entrypoints_relocatable(venv)
    write_install_manifest(recipe_name)
    record = recipe_record(recipe_name)
    try:
        run(
            [
                python,
                "-m",
                "r9700.cli",
                "test",
                "runtime",
                "--profile",
                record["smoke_profile"],
            ]
        )
    except Exception:
        # Never leave a success attestation behind for a failed post-install gate.
        recipe_install_path(recipe_name).unlink(missing_ok=True)
        raise
    print(f"runtime recipe installed: {recipe_install_path(recipe_name)}")


def _prepare_source(name: str, tree: Path, manifest: dict[str, Any]) -> Path:
    spec = manifest["sources"][name]
    full_index_diff = bool(spec.get("full_index_diff", False))
    if not (tree / ".git").is_dir():
        tree.mkdir(parents=True, exist_ok=True)
        run(["git", "init", tree])
        run(["git", "remote", "add", "origin", spec["repository"]], cwd=tree)
    try:
        head = _git(tree, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        head = ""
    if head != spec["commit"]:
        checkout_incomplete = False
        if (
            head
            and tracked_diff_sha256(tree, full_index=full_index_diff)
            != EMPTY_DIFF_SHA256
        ):
            status_lines = _git(tree, "status", "--porcelain").splitlines()
            checkout_incomplete = bool(status_lines) and all(
                line[:2].strip() == "D" for line in status_lines
            )
        if (
            head
            and tracked_diff_sha256(tree, full_index=full_index_diff)
            != EMPTY_DIFF_SHA256
            and not checkout_incomplete
        ):
            raise ConfigurationError(
                f"{name} has local changes and cannot switch to the recipe commit"
            )
        run(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                spec["commit"],
            ],
            cwd=tree,
        )
        checkout = ["git", "checkout", "--detach"]
        if checkout_incomplete:
            checkout.append("--force")
        checkout.append(spec["commit"])
        run(checkout, cwd=tree)
        head = _git(tree, "rev-parse", "HEAD")
    if head != spec["commit"]:
        raise ConfigurationError(f"{name} source HEAD differs from the manifest")
    if name == "aiter":
        # Materialize the pinned source graph before applying our patch image.
        # This makes a fresh installation and an idempotent rerun equivalent.
        _materialize_aiter_submodules(tree)
    current = tracked_diff_sha256(tree, full_index=full_index_diff)
    if current == spec["expected_diff_sha256"]:
        print(f"{name} exact patch image already present")
        return tree
    previous_images = {
        row["sha256"]: int(row["patch_count"])
        for row in spec.get("accepted_previous_images", [])
    }
    if current != EMPTY_DIFF_SHA256 and current not in previous_images:
        raise ConfigurationError(
            f"{name} contains an unsupported tracked diff ({current}); "
            "use a fresh .runtime directory"
        )
    start_index = previous_images.get(current, 0)
    if not 0 <= start_index <= len(spec["patches"]):
        raise ConfigurationError(
            f"{name} append-only patch count is invalid: {start_index}"
        )
    if current in previous_images:
        print(
            f"{name} accepted append-only patch base: {current} "
            f"({start_index} patches)",
            flush=True,
        )
    _apply_patch_image(tree, spec["patches"], start_index=start_index)
    if spec.get("apply_intent_to_add", False):
        # Include patch-created files in the attested diff. Do this separately:
        # `git apply --intent-to-add` on Git 2.43 can replace the index with the
        # patch's path set, making every unaffected upstream file look deleted.
        run(["git", "add", "--intent-to-add", "--", "."], cwd=tree)
    actual = tracked_diff_sha256(tree, full_index=full_index_diff)
    if actual != spec["expected_diff_sha256"]:
        raise ConfigurationError(
            f"{name} post-image mismatch: {actual} != {spec['expected_diff_sha256']}"
        )
    return tree
