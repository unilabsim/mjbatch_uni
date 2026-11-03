# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from mjbatch._bindings import adjoint_info


def test_adjoint_abi_capability_discovery():
  info = adjoint_info()
  assert set(info) == {
    "available",
    "abi_version",
    "capabilities",
    "expected_abi_version",
    "step_vjp_capability",
    "backend",
  }
  assert info["step_vjp_capability"] == 1
  if os.environ.get("MJBATCH_EXPECT_ADJOINT") == "1":
    assert info["backend"] == "adjoint"
    assert info["available"] is True
    assert info["abi_version"] == info["expected_abi_version"] == 1
  elif info["backend"] == "stock":
    assert info["available"] is False
    assert info["abi_version"] == info["capabilities"] == 0
  if os.environ.get("MJBATCH_EXPECT_STEP_VJP") == "1":
    assert info["capabilities"] & info["step_vjp_capability"]
  if os.environ.get("MJBATCH_EXPECT_NO_STEP_VJP") == "1":
    assert not info["capabilities"] & info["step_vjp_capability"]


def test_conflicting_adjoint_expectations(monkeypatch):
  monkeypatch.setenv("MJBATCH_EXPECT_STEP_VJP", "1")
  monkeypatch.setenv("MJBATCH_EXPECT_NO_STEP_VJP", "1")
  if adjoint_info()["available"]:
    with pytest.raises(AssertionError):
      test_adjoint_abi_capability_discovery()
