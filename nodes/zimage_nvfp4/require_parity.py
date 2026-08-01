"""Fail closed if ConvRot act-rotate forward is not armed."""


def require_convrot_parity_forward() -> None:
    """Fail if Linear.forward is not the ConvRot act-rotate wrapper."""
    import comfy.ops

    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward still has HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); quality would be destroyed"
        )
    if not getattr(lin_fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward missing "
            "_hswq_nvfp4_convrot_parity (online act rotation required for "
            "offline W@H^T weights)"
        )
