"""Analysis-only context: decoder sees exactly code vectors, never continuous features."""
from torch.utils.checkpoint import checkpoint

from finetuning.run_index_search import vectors


def analysis_context_forward(model, adapter, features, shape, training=False):
    transformed = adapter(features, shape)
    # Evaluation hooks deliberately replace ST vectors by exact code vectors.
    # Training must retain the native quantizer's ST gradient to the adapter.
    model.quantizer.train(training)
    quantized, vq, indices, perplexity = model.quantizer(transformed)
    if not training:
        quantized = vectors(indices, model.quantizer.embedding.weight, tuple(features.shape[-2:]))
    reconstruction = checkpoint(model.decoder, quantized, use_reentrant=False) if training else model.decoder(quantized)
    return reconstruction[..., :shape[0], :shape[1]], vq, indices, perplexity
