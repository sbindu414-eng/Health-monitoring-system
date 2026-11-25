"""
Real Model-Based Grad-CAM Explainability for Retinal Fundus CNNs.

Generates Gradient-weighted Class Activation Maps (Grad-CAM) directly from the
trained PyTorch CNN's final spatial convolutional block for a target DR grade.
Black background padding outside the circular retinal FOV is masked so the
visual overlay highlights only active retinal regions that influenced the CNN.
"""

import base64
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

GRADCAM_DISCLAIMER = (
    "Explainability Notice: This Grad-CAM heatmap visualizes which retinal regions "
    "most strongly influenced the CNN's internal activation score. It is an interpretation "
    "of model behavior, NOT clinical proof that the highlighted region contains disease."
)


def _resolve_target_conv_layer(model: nn.Module, arch: str) -> Optional[nn.Module]:
    """Return the final spatial convolutional module for Grad-CAM hooking."""
    arch_l = (arch or "").lower()
    if arch_l.startswith(("efficientnet", "mobilenet", "convnext")):
        if hasattr(model, "features") and len(model.features) > 0:
            return model.features[-1]
    if arch_l.startswith("resnet"):
        if hasattr(model, "layer4") and len(model.layer4) > 0:
            return model.layer4[-1]

    # Generic fallback: last Conv2d layer in module hierarchy
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def generate_fundus_gradcam(
    model: nn.Module,
    arch: str,
    input_tensor: torch.Tensor,
    preprocessed_pil: Image.Image,
    target_class: int,
) -> Dict:
    """
    Compute real Grad-CAM heatmap overlay for `target_class` on `preprocessed_pil`.

    Args:
        model: Loaded PyTorch classification model (kept in eval mode).
        arch: Architecture name (e.g. 'efficientnet_b0', 'resnet34').
        input_tensor: Preprocessed 4D tensor (1, 3, H, W).
        preprocessed_pil: Exact RGB image matching `input_tensor` (square FOV-cropped).
        target_class: Class index (0..4) whose activation gradient is explained.
    """
    target_layer = _resolve_target_conv_layer(model, arch)
    if target_layer is None:
        return {
            "available": False,
            "overlay_image": "",
            "disclaimer": GRADCAM_DISCLAIMER,
            "reason": f"Could not resolve spatial convolutional layer for architecture '{arch}'.",
        }

    activations = []
    gradients = []

    def forward_hook(_module, _inp, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_inp, grad_out):
        if grad_out and grad_out[0] is not None:
            gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        x_in = input_tensor.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x_in)
        score = logits[0, int(target_class)]
        score.backward()

        if not activations or not gradients:
            return {
                "available": False,
                "overlay_image": "",
                "disclaimer": GRADCAM_DISCLAIMER,
                "reason": "Grad-CAM hooks captured empty gradients.",
            }

        act = activations[0]  # (1, C, h, w)
        grad = gradients[0]   # (1, C, h, w)

        # Global-average-pool gradients over spatial axes -> channel importance weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * act, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        target_h, target_w = preprocessed_pil.size[1], preprocessed_pil.size[0]
        cam_up = F.interpolate(cam, size=(target_h, target_w), mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].cpu().numpy().astype(np.float32)

        rgb_np = np.asarray(preprocessed_pil.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        fov_mask = gray > 10

        # Normalize strictly inside retinal FOV so dark borders do not skew intensity
        if np.any(fov_mask):
            fov_vals = cam_np[fov_mask]
            c_min = float(np.min(fov_vals))
            c_max = float(np.max(fov_vals))
            if c_max - c_min > 1e-6:
                cam_norm = np.clip((cam_np - c_min) / (c_max - c_min), 0.0, 1.0)
            else:
                cam_norm = np.zeros_like(cam_np)
            cam_norm[~fov_mask] = 0.0
        else:
            c_max = float(np.max(cam_np))
            cam_norm = cam_np / c_max if c_max > 1e-6 else np.zeros_like(cam_np)

        heatmap_uint8 = np.uint8(np.round(cam_norm * 255.0))
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend smoothly inside active retinal FOV; keep outer black margins untouched
        blended = rgb_np.copy()
        alpha = (0.42 * cam_norm[..., None])
        fov_blend = np.uint8(np.clip((1.0 - alpha) * rgb_np + alpha * heatmap_rgb, 0, 255))
        if np.any(fov_mask):
            blended[fov_mask] = fov_blend[fov_mask]
        else:
            blended = fov_blend

        bgr_out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr_out)
        overlay_b64 = f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}" if ok else ""

        # Identify peak focus quadrant inside FOV for text summary
        focus_region = _describe_peak_region(cam_norm, fov_mask)
        return {
            "available": bool(overlay_b64),
            "target_class": int(target_class),
            "target_layer": target_layer.__class__.__name__,
            "focus_region": focus_region,
            "peak_activation_pct": round(float(np.mean(cam_norm[fov_mask] > 0.65) * 100.0), 1) if np.any(fov_mask) else 0.0,
            "overlay_image": overlay_b64,
            "disclaimer": GRADCAM_DISCLAIMER,
        }
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
        model.zero_grad(set_to_none=True)


def _describe_peak_region(cam_norm: np.ndarray, fov_mask: np.ndarray) -> str:
    """Summarize which image quadrant received highest Grad-CAM activation."""
    h, w = cam_norm.shape
    if not np.any(cam_norm > 0.2):
        return "Diffuse / low spatial concentration across retina"
    py, px = np.unravel_index(int(np.argmax(cam_norm)), cam_norm.shape)
    vert = "Superior (Upper)" if py < h * 0.4 else ("Inferior (Lower)" if py > h * 0.6 else "Central Macular")
    horiz = "Left" if px < w * 0.4 else ("Right" if px > w * 0.6 else "Midline")
    return f"{vert} / {horiz} retinal zone"
