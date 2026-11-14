"""Jointly trainable mixed-radix product VQ, with exactly 2048 composite symbols."""
from pathlib import Path
import torch
import torch.nn.functional as F

from finetuning.model_factory import build_model, new_model
from finetuning.run_index_search import nearest, vectors

VERSION = 'factorized-vq-study-v1'
VARIANTS = ('flat', 'product')


def training_check_metadata(rows, root):
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Internal check metadata must contain training samples only')
    result = {Path(root) / row['sample_path']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate internal check samples')
    return result


class ProductEmbedding(torch.nn.Module):
    def __init__(self, left, right, basis, center):
        super().__init__()
        if (left.shape != (32, 64) or right.shape != (64, 64)
                or basis.shape != (128, 128) or center.shape != (128,)
                or any(not torch.isfinite(t).all() for t in (left, right, basis, center))
                or not torch.allclose(basis.T @ basis, torch.eye(128, device=basis.device, dtype=basis.dtype), atol=2e-5, rtol=2e-5)):
            raise ValueError('Expected finite mixed32x64 codebooks and orthogonal full128 basis')
        self.left = torch.nn.Parameter(left.detach().clone())
        self.right = torch.nn.Parameter(right.detach().clone())
        self.register_buffer('basis', basis.detach().clone())
        self.register_buffer('center', center.detach().clone())

    @property
    def weight(self):
        # Canonical packed ID = left_id * 64 + right_id, 11 nominal bits.
        combinations = torch.cat([self.left[:, None, :].expand(32, 64, 64),
                                  self.right[None, :, :].expand(32, 64, 64)], dim=-1).reshape(2048, 128)
        return combinations @ self.basis.T + self.center

    def forward(self, indices):
        return F.embedding(indices, self.weight)


class FactorizedVQCodec(torch.nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown quantizer variant')
        self.base, self.variant = base, variant
        self.num_embeddings = base.num_embeddings
        self.downsample_levels = base.downsample_levels
        self.latent_geometry = base.latent_geometry

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def quantizer(self):
        return self.base.quantizer

    def encode_indices(self, target):
        h, w = target.shape[-2:]
        z = self.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        return nearest(z, self.quantizer.embedding.weight), tuple(z.shape[-2:])

    def decode_indices(self, indices, latent_shape, shape):
        q = vectors(indices, self.quantizer.embedding.weight, latent_shape)
        return self.decoder(q)[..., :shape[0], :shape[1]]

    def forward(self, target):
        return self.base(target)


def initialize_factorized(source, variant, device, product_assets=None):
    base_config = {**source['config'], 'exact_eval_quantization': True, 'activation_checkpointing': True}
    if (base_config.get('architecture', 'native-v1') != 'native-v1'
            or base_config.get('latent_geometry') != 'spatial-4x'
            or base_config['embedding_dim'] != 128 or base_config['num_embeddings'] != 2048
            or base_config['downsample_levels'] != 4):
        raise ValueError('Expected transferred native compact K2048 source')
    base = build_model(base_config, source, device)
    if variant == 'product':
        if product_assets is None:
            raise ValueError('Train-only fitted product assets required')
        base.quantizer.embedding = ProductEmbedding(**{k: product_assets[k].to(device) for k in ('left', 'right', 'basis', 'center')})
    model = FactorizedVQCodec(base, variant).to(device)
    return model, {'architecture': VERSION, 'variant': variant, 'base_config': base_config}


def build_factorized(config, saved, device):
    if config.get('architecture') != VERSION or config != saved['config']:
        raise ValueError('Factorized codec version/configuration differs')
    base_config = config['base_config']
    if (config.get('variant') not in VARIANTS or base_config.get('architecture', 'native-v1') != 'native-v1'
            or base_config.get('latent_geometry') != 'spatial-4x' or base_config.get('embedding_dim') != 128
            or base_config.get('num_embeddings') != 2048 or base_config.get('downsample_levels') != 4):
        raise ValueError('Unsupported factorized codec schema/geometry')
    base = new_model(config['base_config'], device)
    if config['variant'] == 'product':
        base.quantizer.embedding = ProductEmbedding(torch.zeros(32, 64), torch.zeros(64, 64),
                                                    torch.eye(128), torch.zeros(128)).to(device)
    model = FactorizedVQCodec(base, config['variant']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval()


def long_training_gate(initial_product, previous_product, final_product, flat_rmse, product_bytes, flat_bytes, perplexity):
    values = torch.tensor([initial_product, previous_product, final_product, flat_rmse, perplexity], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values <= 0).any() or min(product_bytes, flat_bytes) <= 0:
        return False
    return bool(final_product <= .99 * initial_product and final_product <= .998 * previous_product
                and final_product <= 1.03 * flat_rmse and product_bytes <= 1.05 * flat_bytes and perplexity >= 16)
