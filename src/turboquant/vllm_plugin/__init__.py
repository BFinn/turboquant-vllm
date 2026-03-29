"""
TurboQuant KV cache compression plugin for vLLM.

Compresses KV cache on full-attention layers using TurboQuant's two-stage
quantization (Lloyd-Max MSE + QJL residual correction) and computes attention
directly from compressed data via the asymmetric inner product estimator.
"""

import logging

logger = logging.getLogger(__name__)

_installed = False


def install():
    """Plugin entry point called by vLLM's load_general_plugins().

    Must be re-entrant (safe to call multiple times across processes).
    """
    global _installed
    if _installed:
        return
    _installed = True

    from .config import TurboQuantConfig

    config = TurboQuantConfig.from_env()
    if not config.enabled:
        logger.info("TurboQuant plugin loaded but disabled (VLLM_TURBOQUANT_ENABLED=0)")
        return

    logger.info(
        "TurboQuant plugin: %d-bit compression on layers %s",
        config.bits,
        config.full_attn_layers,
    )

    from .patch import apply_patch

    apply_patch(config)

    logger.info("TurboQuant monkey-patch applied to FlashAttentionImpl")
