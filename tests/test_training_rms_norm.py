from __future__ import annotations

import unittest

import torch

from sensenovalm.model.ops import norm as norm_ops


class TrainingRMSNormTest(unittest.TestCase):
    def test_cpu_path_matches_manual_reference(self) -> None:
        module = norm_ops._RMSNorm(16, eps=1e-6)
        inputs = torch.randn(4, 16, dtype=torch.float32)

        expected = norm_ops.manual_rms_norm(
            inputs,
            module.weight,
            module.normalized_shape,
            module.eps,
            module.add_unit_offset,
        )

        torch.testing.assert_close(module(inputs), expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and norm_ops.flash_rmsnorm_impl,
        "FlashAttention CUDA RMSNorm is unavailable",
    )
    def test_flash_cuda_path_matches_manual_forward_and_backward(self) -> None:
        inputs = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        fused_inputs = inputs.detach().clone().requires_grad_(True)
        fused_weight = weight.detach().clone().requires_grad_(True)

        reference = norm_ops.manual_rms_norm(
            inputs, weight, torch.Size((128,)), 1e-6
        )
        fused = norm_ops._flash_layer_norm_fn(
            fused_inputs,
            fused_weight,
            None,
            eps=1e-6,
            is_rms_norm=True,
        )
        gradient = torch.randn_like(reference)
        reference.backward(gradient)
        fused.backward(gradient)

        torch.testing.assert_close(fused, reference, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            fused_inputs.grad, inputs.grad, rtol=3e-2, atol=3e-2
        )
        # The affine gradient reduces all rows. BF16 Triton and the manual
        # PyTorch graph use different reduction trees, so elementwise relative
        # error is unstable near zero; compare the aggregate error instead.
        relative_weight_error = (
            (fused_weight.grad.float() - weight.grad.float()).norm()
            / weight.grad.float().norm().clamp_min(1e-12)
        )
        self.assertLess(float(relative_weight_error), 5e-3)


if __name__ == "__main__":
    unittest.main()
