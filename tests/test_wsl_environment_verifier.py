from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.verify_wsl_environment import PROFILE_MODULES, verify


class WslEnvironmentVerifierTests(unittest.TestCase):
    @patch("scripts.verify_wsl_environment.platform.release", return_value="microsoft-wsl2")
    @patch(
        "scripts.verify_wsl_environment._probe_module",
        side_effect=lambda name: {"status": "passed", "version": f"test-{name}"},
    )
    def test_minimal_marks_unrequested_integrations_as_skipped(
        self, _probe: object, _release: object
    ) -> None:
        report = verify("minimal")
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"]["modules"]["numpy"]["status"], "passed")
        self.assertEqual(report["checks"]["modules"]["cupy"]["status"], "skipped")

    @patch("scripts.verify_wsl_environment.platform.release", return_value="linux-generic")
    @patch(
        "scripts.verify_wsl_environment._probe_module",
        return_value={"status": "passed", "version": "test"},
    )
    def test_non_wsl_platform_fails_even_when_modules_import(
        self, _probe: object, _release: object
    ) -> None:
        self.assertFalse(verify("minimal")["passed"])

    def test_profiles_are_strictly_additive(self) -> None:
        self.assertLess(set(PROFILE_MODULES["minimal"]), set(PROFILE_MODULES["cpu-full"]))
        self.assertLess(set(PROFILE_MODULES["cpu-full"]), set(PROFILE_MODULES["cuda-full"]))


if __name__ == "__main__":
    unittest.main()
