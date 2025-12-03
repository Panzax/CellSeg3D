"""SwinUNetR wrapper for napari_cellseg3d."""

import inspect

from monai.networks.nets import SwinUNETR

from napari_cellseg3d.utils import LOGGER

logger = LOGGER


class SwinUNETR_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
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

    # def forward(self, x_in):
    #     y = super().forward(x_in)
    # return softmax(y, dim=1)
    # return sigmoid(y)

    # def get_output(self, input):
    #     out = self(input)
    #     return torch.sigmoid(out)

    # def get_validation(self, val_inputs):
    #     return self(val_inputs)

class SwinUNETR_Mlp_LeakyReLU_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_Mlp_LeakyReLU_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
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
class SwinUNETR_SwiGLU_LeakyReLU_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_SwiGLU_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
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
            mlp_type="swiglu",
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

class SwinUNETR_Mlp_ReLUSquared_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_ReLUSquared_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
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
            act_name=("relusquared", {"inplace": True}),
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

class SwinUNETR_SwiGLU_ReLUSquared_(SwinUNETR):
    """SwinUNETR wrapper for napari_cellseg3d."""

    weights_file = "SwinUNetR_SwiGLU_ReLUSquared_latest.pth"
    default_threshold = 0.4

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_img_size=(64, 64, 64),
        use_checkpoint=True,
        **kwargs,
    ):
        """Create a SwinUNetR model.

        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        input_img_size (tuple): input image size
        use_checkpoint (bool): whether to use checkpointing during training.
        **kwargs: additional arguments to SwinUNETR.
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
            mlp_type="swiglu",
            act_name=("relusquared", {"inplace": True}),
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
