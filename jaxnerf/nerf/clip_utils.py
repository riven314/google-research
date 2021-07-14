import math
from typing import Optional
from absl import flags
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from transformers import FlaxCLIPModel

FLAGS = flags.FLAGS
# import jmp
# my_policy = jmp.Policy(compute_dtype=np.float16,
#                        param_dtype=np.float16,
#                        output_dtype=np.float16)


@partial(jax.jit, static_argnums=[0, 1])
def update_semantic_loss(model, clip_model, rng, state, batch, lr):
    # the batch is without shard
    random_rays = batch["random_rays"]
    rng, key_0, key_1 = random.split(rng, 3)

    def semantic_loss(variables):
        # TODO @Alex: (alt) sample less along a ray/ sample on a strided grid (make change on model call)
        # TODO @Alex: (alt) apply mixed precision
        src_ret = model.apply(variables, key_0, key_1, random_rays, False)
        src_image, _, _ = src_ret[-1]
        # reshape flat pixel to an image (assume 3 channels & square shape)
        w = int(math.sqrt(src_image.shape[0]))
        src_image = src_image.reshape([-1, w, w, 3])
        src_image = np.expand_dims(src_image, 0).transpose(0, 3, 1, 2)
        src_image = preprocess_for_CLIP(src_image)
        src_embedding = clip_model.get_image_features(pixel_values=src_image)
        src_embedding /= np.linalg.norm(src_embedding, axis=-1, keepdims=True)
        src_embedding = jnp.array(src_embedding)
        target_embedding = batch["embedding"]
        sc_loss = 0.5 * FLAGS.sc_loss_mult * np.sum((src_embedding - target_embedding) ** 2) / src_embedding.shape[0]
        return sc_loss

    sc_loss, grad = jax.value_and_grad(semantic_loss)(state.optimizer.target)
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(optimizer=new_optimizer)
    return new_state, sc_loss, rng


def trans_t(t):
    return jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=jnp.float32)


def rot_phi(phi):
    return jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(phi), -np.sin(phi), 0],
        [0, jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)


def rot_theta(th):
    return jnp.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, jnp.cos(th), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * jnp.pi) @ c2w
    c2w = rot_theta(theta / 180. * jnp.pi) @ c2w
    c2w = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def random_pose(rng, bds):
    rng, *rng_inputs = jax.random.split(rng, 3)
    radius = random.uniform(rng_inputs[1], minval=bds[0], maxval=bds[1])
    theta = random.uniform(rng_inputs[1], minval=0, maxval=2 * jnp.pi)
    phi = random.uniform(rng_inputs[1], minval=0, maxval=np.pi / 2)
    return pose_spherical(radius, theta, phi)


def preprocess_for_CLIP(image):
    """
    jax-based preprocessing for CLIP
    image  [B, 3, H, W]: batch image
    return [B, 3, 224, 224]: pre-processed image for CLIP
    """
    B, D, H, W = image.shape
    image = jax.image.resize(image, (B, D, 224, 224), 'bicubic')  # assume that images have rectangle shape.
    mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
    std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
    image = (image - mean.astype(image.dtype)) / std.astype(image.dtype)
    return image


# TODO @Alex: VisionModel v.s. original CLIP? (differ by a projection matrix)
def init_CLIP(dtype: str, model_name: Optional[str]) -> FlaxCLIPModel:
    if dtype == 'float16':
        dtype = jnp.float16
    elif dtype == 'float32':
        dtype = jnp.float32
    else:
        raise ValueError

    if model_name is None:
        model_name = 'openai/clip-vit-base-patch32'
    return FlaxCLIPModel.from_pretrained(model_name, dtype=dtype)


# def SC_loss(rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l):
#     """
#     target_emb [1, D]: pre-computed target embedding vector \phi(I)
#     source_img [1, 3, H, W]: source image \hat{I}
#     l: loss weight lambda
#     return: SC_loss
#     """
#     # _,H,W,D = rays.shape
#     rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l = my_policy.cast_to_compute(
#         (rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l))
#     _, H, W, _ = rays.shape
#     source_img = jnp.clip(render_fn(rng_inputs, model, params, None,
#                                    np.reshape(rays, (2, -1, 3)),
#                                    bds[0], bds[1], 1, rand=False),
#                          0, 1)
#     # source_img = np.clip(render_rays(rng_inputs, model, params, None, np.reshape(rays, (2, -1, 3)), bds[0], bds[1], 1, rand=False), 0, 1)
#     source_img = np.reshape(source_img, [1, H, W, 3]).transpose(0, 3, 1, 2)
#     source_img = preprocess_for_CLIP(source_img)
#     source_emb = CLIP_model.get_image_features(pixel_values=source_img)
#     source_emb /= np.linalg.norm(source_emb, axis=-1, keepdims=True)
#     return l/2 * (np.sum((source_emb - target_emb) ** 2) / source_emb.shape[0])

