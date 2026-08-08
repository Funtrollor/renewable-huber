"""Referee for the cross-language CUDA contract.

``native/contracts/rh_cuda_contract.json`` is the single source of truth for the
CUDA ABI.  Four things mirror it -- the C header, the C++ ``static_assert``
translation unit, the Rust ``mod abi`` layout test, and the PyO3 result dicts --
and each mirror used to be free to drift on its own.  These tests diff every
mirror against the manifest.

Two of the mirrors are also checked by a compiler, which is stronger than
anything here: ``native/cuda/src/abi_contract.cpp`` fails the build on a real
layout change, and ``rh-cuda-ffi``'s ``abi_layout`` tests fail ``cargo test``.
The tests below cover what a compiler cannot see -- that the two mirrors, the
header, and the Python key mapping all still describe the *same* contract.

Every parser here asserts how many items it found before comparing anything.
Without that, a regex that silently stops matching turns the whole file into a
suite that always passes.
"""

from __future__ import annotations

import difflib
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).parents[1]
MANIFEST_PATH = REPO / "native" / "contracts" / "rh_cuda_contract.json"
HEADER_PATH = REPO / "native" / "cuda" / "include" / "rh_cuda.h"
CPP_MIRROR_PATH = REPO / "native" / "cuda" / "src" / "abi_contract.cpp"
RUST_CRATE_DIR = REPO / "native" / "crates" / "rh-cuda-ffi" / "src"
# The raw records, the extern block, and the layout tests all live in sys.rs;
# keeping them there is itself part of the contract.
RUST_MIRROR_PATH = RUST_CRATE_DIR / "sys.rs"
PYO3_DIR = REPO / "native" / "crates" / "rh-python-cuda" / "src"
PYO3_PATH = PYO3_DIR / "lib.rs"

MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


class _ContractTestCase(unittest.TestCase):
    """Shared diffing helpers so every failure names both sides."""

    def assert_same_lines(self, expected: list[str], actual: list[str], source: Path) -> None:
        if expected == actual:
            return
        diff = "\n".join(
            difflib.unified_diff(
                expected,
                actual,
                fromfile=_relative(MANIFEST_PATH),
                tofile=_relative(source),
                lineterm="",
            )
        )
        self.fail(
            f"{_relative(source)} no longer matches {_relative(MANIFEST_PATH)}.\n"
            f"Change the manifest first, then every mirror it names.\n\n{diff}"
        )

    def assert_found(self, found: int, expected: int, what: str, source: Path) -> None:
        """Guard against a parser that silently matches nothing."""

        self.assertEqual(
            found,
            expected,
            f"parsed {found} {what} from {_relative(source)} but the manifest declares "
            f"{expected}. Either the mirror is incomplete or this test's parser has "
            f"stopped matching the file's syntax; check the parser before editing "
            f"the manifest.",
        )


class ManifestSelfConsistencyTests(_ContractTestCase):
    """The manifest has to be internally sane before it can referee anything."""

    def test_struct_field_offsets_tile_each_struct_without_overlap(self) -> None:
        for struct, spec in MANIFEST["structs"].items():
            with self.subTest(struct=struct):
                cursor = 0
                for field, offset, width in spec["fields"]:
                    self.assertGreaterEqual(
                        offset, cursor, f"{struct}.{field} overlaps the preceding field"
                    )
                    cursor = offset + width
                self.assertLessEqual(
                    cursor, spec["size"], f"{struct} fields run past its declared size"
                )

    def test_every_public_struct_opens_with_the_version_header(self) -> None:
        # The header's stated rule: every public struct begins with abi_version
        # then struct_size, which is what makes forward-compatible additions
        # detectable instead of misread.
        for struct, spec in MANIFEST["structs"].items():
            with self.subTest(struct=struct):
                names = [field for field, _offset, _width in spec["fields"]]
                self.assertEqual(names[:2], ["abi_version", "struct_size"])


class CHeaderContractTests(_ContractTestCase):
    HEADER = HEADER_PATH.read_text(encoding="utf-8")

    def test_abi_version_matches_manifest(self) -> None:
        match = re.search(r"#define\s+RH_CUDA_ABI_VERSION\s+UINT32_C\((\d+)\)", self.HEADER)
        self.assertIsNotNone(match, "RH_CUDA_ABI_VERSION is no longer declared as expected")
        self.assertEqual(int(match.group(1)), MANIFEST["abi_version"])

    def test_status_dtype_and_flag_constants_match_manifest(self) -> None:
        for prefix, group, pattern in (
            ("RH_CUDA_STATUS_", "status_codes", r"\(\(RhCudaStatus\)(\d+)\)"),
            ("RH_CUDA_DTYPE_", "dtype_codes", r"\(\(RhCudaDType\)(\d+)\)"),
        ):
            found = dict(
                re.findall(
                    rf"#define\s+{prefix}(\w+)\s+{pattern}",
                    self.HEADER,
                )
            )
            self.assert_found(len(found), len(MANIFEST[group]), f"{group} defines", HEADER_PATH)
            for name, value in MANIFEST[group].items():
                self.assertEqual(int(found[name]), value, f"{prefix}{name}")

        # KNOWN_MASK is defined as an or-expression of the other two, so only
        # the single-bit flags parse as literals here.
        flags = dict(
            re.findall(
                r"#define\s+RH_CUDA_ENGINE_FLAG_(\w+)\s+\(UINT64_C\(1\)\s*<<\s*(\d+)\)",
                self.HEADER,
            )
        )
        self.assert_found(len(flags), 2, "single-bit engine flag defines", HEADER_PATH)
        for name, shift in flags.items():
            self.assertEqual(1 << int(shift), MANIFEST["engine_flags"][name])

    def test_struct_field_names_and_order_match_manifest(self) -> None:
        bodies = dict(
            (name, body)
            for body, name in re.findall(
                r"typedef struct \w+ \{(.*?)\}\s*(\w+);", self.HEADER, re.DOTALL
            )
        )
        self.assert_found(
            len(bodies), len(MANIFEST["structs"]), "public struct definitions", HEADER_PATH
        )
        for struct, spec in MANIFEST["structs"].items():
            with self.subTest(struct=struct):
                declarations = re.findall(
                    r"^\s*(?:const\s+)?[\w*]+(?:\s*\*)?\s*(\w+)\s*;\s*(?:/\*.*)?$",
                    bodies[struct],
                    re.MULTILINE,
                )
                self.assert_same_lines(
                    [field for field, _offset, _width in spec["fields"]],
                    declarations,
                    HEADER_PATH,
                )

    def test_exported_functions_match_manifest(self) -> None:
        functions = re.findall(
            r"^RH_CUDA_API\s+[^(;]*?(\brh_cuda_\w+)\s*\(", self.HEADER, re.MULTILINE
        )
        self.assert_found(
            len(functions), len(MANIFEST["functions"]), "RH_CUDA_API declarations", HEADER_PATH
        )
        self.assert_same_lines(MANIFEST["functions"], functions, HEADER_PATH)

    def test_both_batch_structs_document_the_narrow_width(self) -> None:
        # Pins the fix for the drift this contract work started from: the header
        # used to claim intercept construction always belonged to the caller.
        for struct in MANIFEST["batch_column_contract"]["applies_to"]:
            with self.subTest(struct=struct):
                index = self.HEADER.index(f"}} {struct};")
                comment_start = self.HEADER.rindex("/*", 0, self.HEADER.rindex("typedef", 0, index))
                comment = self.HEADER[comment_start:index]
                self.assertIn("n_features_in", comment)
                self.assertIn("n_parameters", comment)


class CppMirrorTests(_ContractTestCase):
    SOURCE = CPP_MIRROR_PATH.read_text(encoding="utf-8")

    def test_struct_sizes_and_offsets_are_asserted(self) -> None:
        sizes = dict(re.findall(r"static_assert\(sizeof\((\w+)\) == (\d+),", self.SOURCE))
        offsets = dict(
            ((struct, field), int(offset))
            for struct, field, offset in re.findall(
                r"static_assert\(offsetof\((\w+), (\w+)\) == (\d+),", self.SOURCE
            )
        )
        expected_offsets = {
            (struct, field): offset
            for struct, spec in MANIFEST["structs"].items()
            for field, offset, _width in spec["fields"]
        }
        self.assert_found(
            len(offsets), len(expected_offsets), "offsetof assertions", CPP_MIRROR_PATH
        )
        # sizeof(void*) and the per-field sizeof(((S*)0)->f) forms do not match
        # this pattern, so only the struct sizes land here.
        self.assert_found(
            len(sizes), len(MANIFEST["structs"]), "struct sizeof assertions", CPP_MIRROR_PATH
        )
        for struct, spec in MANIFEST["structs"].items():
            self.assertEqual(int(sizes[struct]), spec["size"], f"sizeof({struct})")
        self.assertEqual(offsets, expected_offsets)

    def test_field_widths_are_asserted(self) -> None:
        widths = {
            (struct, field): int(width)
            for struct, field, width in re.findall(
                r"static_assert\(sizeof\(\(\((\w+)\*\)0\)->(\w+)\) == (\d+),", self.SOURCE
            )
        }
        expected = {
            (struct, field): width
            for struct, spec in MANIFEST["structs"].items()
            for field, _offset, width in spec["fields"]
        }
        self.assert_found(len(widths), len(expected), "field width assertions", CPP_MIRROR_PATH)
        self.assertEqual(widths, expected)

    def test_pointer_size_and_abi_version_are_asserted(self) -> None:
        self.assertIn(
            f"static_assert(sizeof(void*) == {MANIFEST['pointer_size']},",
            self.SOURCE,
        )
        self.assertIn(
            f"static_assert(RH_CUDA_ABI_VERSION == {MANIFEST['abi_version']}u,",
            self.SOURCE,
        )

    def test_mirror_defines_no_symbols(self) -> None:
        # It must stay assertion-only: adding code here would make a link
        # failure, not a clear compile failure, the way drift surfaces.
        self.assertNotIn("int main", self.SOURCE)
        stripped = re.sub(r"/\*.*?\*/", "", self.SOURCE, flags=re.DOTALL)
        for line in stripped.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "//", "static_assert", '"', ")")):
                continue
            self.fail(f"{_relative(CPP_MIRROR_PATH)} should only hold static_asserts: {line!r}")


class RustMirrorTests(_ContractTestCase):
    SOURCE = RUST_MIRROR_PATH.read_text(encoding="utf-8")

    def test_repr_c_records_are_not_gated_on_the_cuda_feature(self) -> None:
        # This is what lets `cargo test -p rh-cuda-ffi` verify layout on a
        # runner with no CUDA toolchain. If mod abi ever moves back behind the
        # feature gate, the layout tests stop running in ordinary CI and the
        # Rust mirror silently goes unchecked.
        match = re.search(r"^(.*)\n(?:pub\(crate\) )?mod abi \{", self.SOURCE, re.MULTILINE)
        self.assertIsNotNone(match, "mod abi is no longer declared as expected")
        self.assertNotIn('#[cfg(feature = "cuda")]', match.group(1).splitlines()[-1])
        self.assertRegex(self.SOURCE, r'#\[cfg\(feature = "cuda"\)\]\n(?:pub\(crate\) )?mod ffi \{')

    def test_struct_sizes_and_offsets_are_asserted(self) -> None:
        sizes = dict(
            (struct, int(size))
            for struct, size in re.findall(
                r"assert_eq!\(size_of::<(\w+)>\(\), (\d+)\);", self.SOURCE
            )
        )
        offsets = dict(
            ((struct, field), int(offset))
            for struct, field, offset in re.findall(
                r"assert_eq!\(offset_of!\((\w+), (\w+)\), (\d+)\);", self.SOURCE
            )
        )
        expected_offsets = {
            (struct, field): offset
            for struct, spec in MANIFEST["structs"].items()
            for field, offset, _width in spec["fields"]
        }
        self.assert_found(
            len(offsets), len(expected_offsets), "offset_of! assertions", RUST_MIRROR_PATH
        )
        self.assert_found(
            len(sizes), len(MANIFEST["structs"]), "size_of assertions", RUST_MIRROR_PATH
        )
        for struct, spec in MANIFEST["structs"].items():
            self.assertEqual(sizes[struct], spec["size"], f"size_of::<{struct}>()")
        self.assertEqual(offsets, expected_offsets)

    def test_repr_c_field_names_and_order_match_manifest(self) -> None:
        bodies = dict(re.findall(r"pub struct (\w+) \{(.*?)\n    \}", self.SOURCE, re.DOTALL))
        for struct, spec in MANIFEST["structs"].items():
            with self.subTest(struct=struct):
                self.assertIn(struct, bodies, f"{struct} is missing from the Rust mirror")
                fields = re.findall(r"pub (\w+):", bodies[struct])
                self.assert_same_lines(
                    [field for field, _offset, _width in spec["fields"]],
                    fields,
                    RUST_MIRROR_PATH,
                )

    def test_abi_version_is_queried_from_the_linked_library(self) -> None:
        # A source-level mirror cannot catch a stale binary; only asking the
        # library can. The declaration, the check and its call site sit in
        # three different modules, so this one looks at the whole crate.
        crate = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(RUST_CRATE_DIR.rglob("*.rs"))
        )
        for fragment in (
            "pub fn rh_cuda_abi_version() -> u32;",
            "fn linked_abi_version_matches()",
            "linked_abi_version_matches()?;",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, crate)


class PyO3ResultKeyTests(_ContractTestCase):
    SOURCE = PYO3_PATH.read_text(encoding="utf-8")

    def _keys_in(self, function: str) -> list[str]:
        """Return the result-dict keys one function sets, in first-set order.

        The body runs from the ``fn`` line to the next ``}`` at the same
        indentation, which handles both free functions and ``#[pymethods]``
        members, and signatures that rustfmt has spread over several lines.
        Keys are de-duplicated because ``version`` sets the same three inside
        both arms of a match.
        """

        match = re.search(rf"^([ ]*)fn {function}\b", self.SOURCE, re.MULTILINE)
        self.assertIsNotNone(match, f"{function} is no longer defined as expected")
        indent = match.group(1)
        end = self.SOURCE.index(f"\n{indent}}}\n", match.end())
        body = self.SOURCE[match.end() : end]

        keys: list[str] = []
        for key in re.findall(r'result\.set_item\(\s*"(\w+)"', body):
            if key not in keys:
                keys.append(key)
        self.assertTrue(keys, f"{function} no longer sets any result keys")
        return keys

    def test_state_dict_keys_match_manifest(self) -> None:
        self.assert_same_lines(
            MANIFEST["python_result_keys"]["state"],
            self._keys_in("state_dict_from_parts"),
            PYO3_PATH,
        )

    def test_diagnostics_keys_match_manifest(self) -> None:
        self.assert_same_lines(
            sorted(MANIFEST["python_result_keys"]["diagnostics"]),
            sorted(self._keys_in("add_diagnostics")),
            PYO3_PATH,
        )

    def test_update_paths_add_their_declared_extra_keys(self) -> None:
        for function, group in (
            ("update_typed", "update_extra"),
            ("update_device_typed", "update_device_extra"),
        ):
            with self.subTest(function=function):
                self.assert_same_lines(
                    sorted(MANIFEST["python_result_keys"][group]),
                    sorted(self._keys_in(function)),
                    PYO3_PATH,
                )

    def test_version_keys_match_manifest(self) -> None:
        self.assert_same_lines(
            sorted(MANIFEST["python_result_keys"]["version"]),
            sorted(self._keys_in("version")),
            PYO3_PATH,
        )

    def test_engine_feature_keys_match_manifest(self) -> None:
        self.assert_same_lines(
            sorted(MANIFEST["python_result_keys"]["features"]),
            sorted(self._keys_in("features")),
            PYO3_PATH,
        )

    def test_python_api_version_matches_manifest(self) -> None:
        match = re.search(r"const PYTHON_API_VERSION: u32 = (\d+);", self.SOURCE)
        self.assertIsNotNone(match, "PYTHON_API_VERSION is no longer declared as expected")
        self.assertEqual(int(match.group(1)), MANIFEST["python_api_version"])


class PythonBackendConsumesManifestKeysTests(_ContractTestCase):
    """Close the loop: prove the Python adapter reads what the PyO3 layer writes."""

    def _result(self) -> dict[str, object]:
        import numpy as np

        keys = MANIFEST["python_result_keys"]
        result: dict[str, object] = {
            "coefficients": np.zeros(2, dtype=np.float64),
            "information": np.eye(2, dtype=np.float64),
            "n_samples_seen": 4,
            "batch_count": 1,
            "previous_lambda": 0.0,
            "weight_sum": 4.0,
            "iterations": 3,
            "converged": True,
            "used_regularized_fallback": False,
            "objective": 1.5,
            "lambda_value": 0.25,
            "bandwidth": 2.0,
            "state_is_detached": True,
        }
        expected = set(keys["state"]) | set(keys["diagnostics"]) | set(keys["update_extra"])
        self.assertEqual(
            set(result),
            expected,
            "this fixture drifted from the manifest's key sets; update it alongside them",
        )
        return result

    def _decode(self, result: dict[str, object]):
        import numpy as np

        from renewable_huber.backends.native_cuda_backend import NativeCudaBackend
        from renewable_huber.state import RenewableHuberState

        previous = RenewableHuberState.empty(1, fit_intercept=True, xp=np, dtype=np.float64)
        # __init__ loads the extension and probes the device; the decoder itself
        # needs neither, so bind it to a bare instance.
        backend = NativeCudaBackend.__new__(NativeCudaBackend)
        backend.dtype = np.dtype(np.float64)
        return NativeCudaBackend._decode_result(backend, result, previous)

    def test_manifest_keys_are_sufficient_to_decode_a_result(self) -> None:
        state, diagnostics = self._decode(self._result())
        self.assertEqual(state.n_samples_seen, 4)
        self.assertEqual(state.batch_count, 1)
        self.assertEqual(state.weight_sum, 4.0)
        self.assertEqual(diagnostics.iterations, 3)
        self.assertTrue(diagnostics.converged)
        self.assertEqual(diagnostics.bandwidth, 2.0)

    def test_every_required_key_is_actually_required(self) -> None:
        # Without this, a key could be dropped from the PyO3 side and the
        # manifest and stay green as long as nothing read it.
        keys = MANIFEST["python_result_keys"]
        required = set(keys["state"]) | set(keys["diagnostics"])
        for key in sorted(required):
            with self.subTest(key=key):
                result = self._result()
                del result[key]
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    self._decode(result)

    def test_backend_version_gate_matches_manifest(self) -> None:
        from renewable_huber.backends import native_cuda_backend

        self.assertEqual(native_cuda_backend._EXPECTED_ABI_VERSION, MANIFEST["abi_version"])
        self.assertEqual(
            native_cuda_backend._EXPECTED_PYTHON_API_VERSION,
            MANIFEST["python_api_version"],
        )


if __name__ == "__main__":
    unittest.main()
