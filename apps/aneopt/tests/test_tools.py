"""Tools that need no Neural Engine."""
import pytest
import torch

from aneopt.audit import audit_package
from aneopt.export import convert
from aneopt.layers import SplitMLP, conv1x1
from aneopt.measure import hist_stats
from aneopt.probe import find_step


class Wide(torch.nn.Module):
    def __init__(self, split):
        super().__init__()
        fw, fb, pw, pb = torch.randn(4096, 1024), torch.randn(4096), torch.randn(1024, 4096), torch.randn(1024)
        self.mlp = SplitMLP(fw, fb, pw, pb) if split else torch.nn.Sequential(conv1x1(fw, fb), torch.nn.GELU(), conv1x1(pw, pb))

    def forward(self, x):
        return x + self.mlp(x)


@pytest.mark.parametrize("split,expected", [(False, 1), (True, 0)])
def test_audit_finds_wide_activation_only_before_split(tmp_path, split, expected):
    path = tmp_path / "m.mlpackage"
    convert(Wide(split), torch.rand(1, 1024, 1, 577), path)
    result = audit_package(path)
    wide = [a for a in result.over() if a.shape == (1, 4096, 1, 577)]
    assert len(result.over()) >= expected and bool(wide) == bool(expected)
    if wide:
        assert wide[0].bytes == 2 * 4096 * 577 and wide[0].padded_bytes == 4096 * 1216
    assert 16e6 < result.weight_bytes < 18e6  # two 1024 x 4096 FP16 matrices are 16.8 MB


def test_hist_stats():
    stats = hist_stats({"  8GB/s": 1, " 60GB/s": 3, "other": 5})
    assert stats["p50"] == 60 and stats["max_bucket"] == 60 and stats["over_30_gb_s"] == 0.75


def test_find_step_uses_measured_sweep():
    rows = [dict(c=256, activ_kb=1792, tmac_s=2.41), dict(c=256, activ_kb=2048, tmac_s=1.57), dict(c=256, activ_kb=2560, tmac_s=1.59)]
    assert find_step(rows)["first_slow_kb"] == 2048
    assert find_step(rows[1:]) is None
    coarse = [dict(c=1024, activ_kb=1154, tmac_s=3.85), dict(c=1024, activ_kb=2048, tmac_s=3.08)]
    assert find_step(coarse + rows)["channels"] == 256
