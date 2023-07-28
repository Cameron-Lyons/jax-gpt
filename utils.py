"""Utilities for loading GPT-2 models from TF ckpt files."""
import json
import os
import re
from typing import Literal, Dict, Any
import jax.numpy as jnp
import requests
import tensorflow as tf
from tqdm import tqdm
from encoder import get_encoder


def download_gpt2_files(
    model_size: Literal["124M", "355M", "774M", "1558M"], model_dir: str
):
    """Download the GPT-2 model files from OpenAI"""
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True, timeout=10)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb", encoding="utf-8") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(
    tf_ckpt_path: str, hparams: Dict[str, Any]
) -> Dict[str, Any]:
    """Load GPT-2 params from a TF checkpoint"""

    def set_in_nested_dict(d, keys, val):
        """Set a value in a nested dict using a list of keys"""
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = jnp.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/"):]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(
    model_size: Literal["124M", "355M", "774M", "1558M"], models_dir: str
):
    """Load the encoder, hparams, and params for a given model size"""
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    with open(os.path.join(model_dir, "hparams.json"), encoding="utf-8") as f:
        hparams = json.load(f)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return encoder, hparams, params
