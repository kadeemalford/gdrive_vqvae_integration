"""Local, versioned architecture factory; never infer architecture from weight shapes."""
from torch import nn

try:
    from .latent_geometry import apply_geometry, checkpoint_geometry
except ImportError:
    from latent_geometry import apply_geometry, checkpoint_geometry

ARCHITECTURES = ("native-v1", "temporal-dilated-v1")


def architecture(config):
    name = config.get("architecture", "native-v1")
    if name not in ARCHITECTURES:
        raise ValueError("Unknown checkpoint architecture")
    return name


def temporal_residuals(model):
    """Replace second residual 3x3 by 1x9, preserving parameter count and grid.

    Dilations 1/2/4 repeat within encoder and decoder separately. No skip bypasses
    quantization; spatial mixing remains in the first 3x3 of every residual block.
    """
    from vqvae.train_vqvae import ResidualBlock
    for module in (model.encoder, model.decoder):
        blocks = [block for block in module.modules() if isinstance(block, ResidualBlock)]
        if not blocks:
            raise ValueError("Unsupported residual topology")
        for index, block in enumerate(blocks):
            old = block.net[-1]
            if not isinstance(old, nn.Conv2d) or old.kernel_size != (3, 3):
                raise ValueError("Unsupported residual convolution")
            dilation = (1, 2, 4)[index % 3]
            block.net[-1] = nn.Conv2d(old.in_channels, old.out_channels, (1, 9),
                                     padding=(0, 4 * dilation), dilation=(1, dilation),
                                     bias=old.bias is not None)
    return model


def new_model(config, device):
    from vqvae.train_vqvae import VQVAE
    model = VQVAE(in_channels=1, base_channels=int(config["base_channels"]),
                  embedding_dim=int(config["embedding_dim"]), num_embeddings=int(config["num_embeddings"]),
                  downsample_levels=int(config["downsample_levels"]),
                  commitment_cost=float(config.get("commitment_cost", .25)))
    if architecture(config) == "temporal-dilated-v1":
        temporal_residuals(model)
    apply_geometry(model, checkpoint_geometry({"config": config}))
    return model.to(device)


def build_model(config, checkpoint, device):
    if architecture(config) != architecture(checkpoint.get("config", {})):
        raise ValueError("Conflicting checkpoint architecture")
    if architecture(config) == "native-v1":
        from vqvae.evaluate_vqvae import build_model as native_model
        model = native_model(config, checkpoint, device)
    else:
        model = new_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    apply_geometry(model, checkpoint_geometry(checkpoint))
    return model.eval()
