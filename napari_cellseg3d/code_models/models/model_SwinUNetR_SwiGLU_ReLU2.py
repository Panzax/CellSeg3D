"""SwinUNetR_SwiGLU_ReLU2 wrapper for napari_cellseg3d."""

import inspect
import sys
from pathlib import Path

# Add parent directories to path to import from finetune module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

try:
    from cell_observatory_finetune.models.meta_arch.swin_unetr_swiglu_relu2 import SwinUNETR_SwiGLU_ReLU2
except ImportError as e:
    # Fallback: try importing from monai if finetune module is not available
    # In this case, we'd need to have the variant in MONAI submodule
    raise ImportError(
        f"Could not import SwinUNETR_SwiGLU_ReLU2 from cell_observatory_finetune: {e}. "
        "Make sure the finetune module is available in the Python path."
    ) from e

from napari_cellseg3d.utils import LOGGER

logger = LOGGER


class SwinUNETR_SwiGLU_ReLU2_(SwinUNETR_SwiGLU_ReLU2):
    """SwinUNETR_SwiGLU_ReLU2 wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_SwiGLU_ReLU2_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR_SwiGLU_ReLU2 model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR_SwiGLU_ReLU2.
        """
        parent_init = super().__init__
        sig = inspect.signature(parent_init)
        init_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=use_checkpoint,
            feature_size=48,
            drop_rate=0.5,
            attn_drop_rate=0.5,
            use_v2=True,
            **kwargs,
        )
        if "img_size" in sig.parameters:
            # since MONAI API changes depending on py3.8 or py3.9
            init_kwargs["img_size"] = input_img_size
        if "dropout_prob" in kwargs:
            init_kwargs["drop_rate"] = kwargs["dropout_prob"]
            init_kwargs.pop("dropout_prob")
        try:
            parent_init(**init_kwargs)
        except TypeError as e:
            logger.warning(f"Caught TypeError: {e}")
            init_kwargs["in_channels"] = 1
            parent_init(**init_kwargs)

