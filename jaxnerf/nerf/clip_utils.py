import jax
from jax import random
import jax.numpy as np
import jmp
my_policy = jmp.Policy(compute_dtype=np.float16,
                       param_dtype=np.float16,
                       output_dtype=np.float16)


def trans_t(t):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_phi(phi):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_theta(th):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def random_pose(rng, bds):
    rng, *rng_inputs = jax.random.split(rng, 3)
    radius = random.uniform(rng_inputs[1], minval=bds[0], maxval=bds[1])
    theta = random.uniform(rng_inputs[1], minval=0, maxval=2 * np.pi)
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
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
    image = (image - mean.astype(image.dtype)) / std.astype(image.dtype)
    return image


def SC_loss(rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l):
    """
    target_emb [1, D]: pre-computed target embedding vector \phi(I)
    source_img [1, 3, H, W]: source image \hat{I}
    l: loss weight lambda
    return: SC_loss
    """
    # _,H,W,D = rays.shape
    rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l = my_policy.cast_to_compute(
        (rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l))
    _, H, W, _ = rays.shape
    source_img = np.clip(render_fn(rng_inputs, model, params, None,
                                   np.reshape(rays, (2, -1, 3)),
                                   bds[0], bds[1], 1, rand=False),
                         0, 1)
    # source_img = np.clip(render_rays(rng_inputs, model, params, None, np.reshape(rays, (2, -1, 3)), bds[0], bds[1], 1, rand=False), 0, 1)
    source_img = np.reshape(source_img, [1, H, W, 3]).transpose(0, 3, 1, 2)
    source_img = preprocess_for_CLIP(source_img)
    source_emb = CLIP_model.get_image_features(pixel_values=source_img)
    source_emb /= np.linalg.norm(source_emb, axis=-1, keepdims=True)
    return l/2 * (np.sum((source_emb - target_emb) ** 2) / source_emb.shape[0])
