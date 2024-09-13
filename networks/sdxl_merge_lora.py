import math
import argparse
import itertools
import json
import os
import re
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, sdxl_model_util, train_util
import library.model_util as model_util
import lora
import oft
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)
import concurrent.futures

ACCEPTABLE = [12, 17, 20, 26]
SDXL_LAYER_NUM = [12, 20]

LAYER12 = {
    "BASE": True,
    "IN00": False, "IN01": False, "IN02": False, "IN03": False, "IN04": True, "IN05": True,
    "IN06": False, "IN07": True, "IN08": True, "IN09": False, "IN10": False, "IN11": False,
    "MID": True,
    "OUT00": True, "OUT01": True, "OUT02": True, "OUT03": True, "OUT04": True, "OUT05": True,
    "OUT06": False, "OUT07": False, "OUT08": False, "OUT09": False, "OUT10": False, "OUT11": False
}

LAYER17 = {
    "BASE": True,
    "IN00": False, "IN01": True, "IN02": True, "IN03": False, "IN04": True, "IN05": True,
    "IN06": False, "IN07": True, "IN08": True, "IN09": False, "IN10": False, "IN11": False,
    "MID": True,
    "OUT00": False, "OUT01": False, "OUT02": False, "OUT03": True, "OUT04": True, "OUT05": True,
    "OUT06": True, "OUT07": True, "OUT08": True, "OUT09": True, "OUT10": True, "OUT11": True,
}

LAYER20 = {
    "BASE": True,
    "IN00": True, "IN01": True, "IN02": True, "IN03": True, "IN04": True, "IN05": True,
    "IN06": True, "IN07": True, "IN08": True, "IN09": False, "IN10": False, "IN11": False,
    "MID": True,
    "OUT00": True, "OUT01": True, "OUT02": True, "OUT03": True, "OUT04": True, "OUT05": True,
    "OUT06": True, "OUT07": True, "OUT08": True, "OUT09": False, "OUT10": False, "OUT11": False,
}

LAYER26 = {
    "BASE": True,
    "IN00": True, "IN01": True, "IN02": True, "IN03": True, "IN04": True, "IN05": True,
    "IN06": True, "IN07": True, "IN08": True, "IN09": True, "IN10": True, "IN11": True,
    "MID": True,
    "OUT00": True, "OUT01": True, "OUT02": True, "OUT03": True, "OUT04": True, "OUT05": True,
    "OUT06": True, "OUT07": True, "OUT08": True, "OUT09": True, "OUT10": True, "OUT11": True,
}

assert len([v for v in LAYER12.values() if v]) == 12
assert len([v for v in LAYER17.values() if v]) == 17
assert len([v for v in LAYER20.values() if v]) == 20
assert len([v for v in LAYER26.values() if v]) == 26

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


def get_lbw_block_index(lora_name: str, is_sdxl: bool = False) -> int:
    # lbw block index is 0-based, but 0 for text encoder, so we return 0 for text encoder
    if "text_model_encoder_" in lora_name:  # LoRA for text encoder
        return 0

    # lbw block index is 1-based for U-Net, and no "input_blocks.0" in CompVis SD, so "input_blocks.1" have index 2
    block_idx = -1  # invalid lora name
    if not is_sdxl:
        NUM_OF_BLOCKS = 12  # up/down blocks
        m = RE_UPDOWN.search(lora_name)
        if m:
            g = m.groups()
            up_down = g[0]
            i = int(g[1])
            j = int(g[3])
            if up_down == "down":
                if g[2] == "resnets" or g[2] == "attentions":
                    idx = 3 * i + j + 1
                elif g[2] == "downsamplers":
                    idx = 3 * (i + 1)
                else:
                    return block_idx  # invalid lora name
            elif up_down == "up":
                if g[2] == "resnets" or g[2] == "attentions":
                    idx = 3 * i + j
                elif g[2] == "upsamplers":
                    idx = 3 * i + 2
                else:
                    return block_idx  # invalid lora name

            if g[0] == "down":
                block_idx = 1 + idx  # 1-based index, down block index
            elif g[0] == "up":
                block_idx = 1 + NUM_OF_BLOCKS + 1 + idx  # 1-based index, num blocks, mid block, up block index

        elif "mid_block_" in lora_name:
            block_idx = 1 + NUM_OF_BLOCKS  # 1-based index, num blocks, mid block
    else:
        if lora_name.startswith("lora_unet_"):
            name = lora_name[len("lora_unet_") :]
            if name.startswith("time_embed_") or name.startswith("label_emb_"):  # 1, No LoRA in sd-scripts
                block_idx = 1
            elif name.startswith("input_blocks_"):  # 1-8 to 2-9
                block_idx = 1 + int(name.split("_")[2])
            elif name.startswith("middle_block_"):  # 10
                block_idx = 10
            elif name.startswith("output_blocks_"):  # 0-8 to 11-19
                block_idx = 11 + int(name.split("_")[2])
            elif name.startswith("out_"):  # 20, No LoRA in sd-scripts
                block_idx = 20

    return block_idx


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, model, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)


def detect_method_from_training_model(models, dtype):
    for model in models:
        lora_sd, _ = load_state_dict(model, dtype)
        for key in tqdm(lora_sd.keys()):
            if "lora_up" in key or "lora_down" in key:
                return "LoRA"
            elif "oft_blocks" in key:
                return "OFT"


def merge_to_sd_model(text_encoder1, text_encoder2, unet, models, ratios, lbws, merge_dtype):
    text_encoder1.to(merge_dtype)
    text_encoder1.to(merge_dtype)
    unet.to(merge_dtype)

    # detect the method: OFT or LoRA_module
    method = detect_method_from_training_model(models, merge_dtype)
    logger.info(f"method:{method}")

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder1, text_encoder2, unet]):
        if method == "LoRA":
            if i <= 1:
                if i == 0:
                    prefix = lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER1
                else:
                    prefix = lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER2
                target_replace_modules = lora.LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            else:
                prefix = lora.LoRANetwork.LORA_PREFIX_UNET
                target_replace_modules = (
                    lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE + lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
                )
        elif method == "OFT":
            prefix = oft.OFTNetwork.OFT_PREFIX_UNET
            # ALL_LINEAR includes ATTN_ONLY, so we don't need to specify ATTN_ONLY
            target_replace_modules = (
                oft.OFTNetwork.UNET_TARGET_REPLACE_MODULE_ALL_LINEAR + oft.OFTNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    if lbws and (method == "LoRA"):
        try:
            # lbwは"[1,1,1,1,1,1,1,1,1,1,1,1]"のような文字列で与えられることを期待している
            lbws = [json.loads(lbw) for lbw in lbws]
        except Exception:
            raise ValueError(f"format of lbws are must be json / 層別適用率はJSON形式で書いてください")
        assert all(isinstance(lbw, list) for lbw in lbws), f"lbws are must be list / 層別適用率はリストにしてください"
        assert len(set(len(lbw) for lbw in lbws)) == 1, "all lbws should have the same length  / 層別適用率は同じ長さにしてください"
        assert all(len(lbw) in ACCEPTABLE for lbw in lbws), f"length of lbw are must be in {ACCEPTABLE} / 層別適用率の長さは{ACCEPTABLE}のいずれかにしてください"
        assert all(all(isinstance(weight, (int, float)) for weight in lbw) for lbw in lbws), f"values of lbs are must be numbers / 層別適用率の値はすべて数値にしてください"

        layer_num = len(lbws[0])
        is_sdxl = True if layer_num in SDXL_LAYER_NUM else False
        FLAGS = {
            "12": LAYER12.values(),
            "17": LAYER17.values(),
            "20": LAYER20.values(),
            "26": LAYER26.values(),
        }[str(layer_num)]
        LBW_TARGET_IDX = [i for i, flag in enumerate(FLAGS) if flag]

    if lbws and (method == "OFT"):
        raise NotImplementedError(f"OFT does not support LBW. / OFTではLBWはサポートされていません")

    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        logger.info(f"merging...")

        if method == "LoRA":
            if lbw:
                lbw_weights = [1] * 26
                for index, value in zip(LBW_TARGET_IDX, lbw):
                    lbw_weights[index] = value
                print(dict(zip(LAYER26.keys(), lbw_weights)))

            for key in tqdm(lora_sd.keys()):
                if "lora_down" in key:
                    up_key = key.replace("lora_down", "lora_up")
                    alpha_key = key[: key.index("lora_down")] + "alpha"

                    # find original module for this lora
                    module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                    if module_name not in name_to_module:
                        logger.info(f"no module found for LoRA weight: {key}")
                        continue
                    module = name_to_module[module_name]
                    # logger.info(f"apply {key} to {module}")

                    down_weight = lora_sd[key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    if lbw:
                        index = get_lbw_block_index(key, is_sdxl)
                        is_lbw_target = index in LBW_TARGET_IDX
                        if is_lbw_target:
                            scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

                    # W <- W + U * D
                    weight = module.weight
                    # logger.info(module_name, down_weight.size(), up_weight.size())
                    if len(weight.size()) == 2:
                        # linear
                        weight = weight + ratio * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        weight = (
                            weight
                            + ratio
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                        weight = weight + ratio * conved * scale

                    module.weight = torch.nn.Parameter(weight)

        elif method == "OFT":

            multiplier = 1.0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for key in tqdm(lora_sd.keys()):
                if "oft_blocks" in key:
                    oft_blocks = lora_sd[key]
                    dim = oft_blocks.shape[0]
                    break
            for key in tqdm(lora_sd.keys()):
                if "alpha" in key:
                    oft_blocks = lora_sd[key]
                    alpha = oft_blocks.item()
                    break

            def merge_to(key):
                if "alpha" in key:
                    return

                # find original module for this OFT
                module_name = ".".join(key.split(".")[:-1])
                if module_name not in name_to_module:
                    logger.info(f"no module found for OFT weight: {key}")
                    return
                module = name_to_module[module_name]

                # logger.info(f"apply {key} to {module}")

                oft_blocks = lora_sd[key]

                if isinstance(module, torch.nn.Linear):
                    out_dim = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    out_dim = module.out_channels

                num_blocks = dim
                block_size = out_dim // dim
                constraint = (0 if alpha is None else alpha) * out_dim

                block_Q = oft_blocks - oft_blocks.transpose(1, 2)
                norm_Q = torch.norm(block_Q.flatten())
                new_norm_Q = torch.clamp(norm_Q, max=constraint)
                block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
                I = torch.eye(block_size, device=oft_blocks.device).unsqueeze(0).repeat(num_blocks, 1, 1)
                block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())
                block_R_weighted = multiplier * block_R + (1 - multiplier) * I
                R = torch.block_diag(*block_R_weighted)

                # get org weight
                org_sd = module.state_dict()
                org_weight = org_sd["weight"].to(device)

                R = R.to(org_weight.device, dtype=org_weight.dtype)

                if org_weight.dim() == 4:
                    weight = torch.einsum("oihw, op -> pihw", org_weight, R)
                else:
                    weight = torch.einsum("oi, op -> pi", org_weight, R)

                weight = weight.contiguous()  # Make Tensor contiguous; required due to ThreadPoolExecutor

                module.weight = torch.nn.Parameter(weight)

            # TODO multi-threading may cause OOM on CPU if cpu_count is too high and RAM is not enough
            max_workers = 1 if device.type != "cpu" else None  # avoid OOM on GPU
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(merge_to, lora_sd.keys()), total=len(lora_sd.keys())))


def merge_lora_models(models, ratios, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    v2 = None
    base_model = None

    if lbws:
        try:
            # lbwは"[1,1,1,1,1,1,1,1,1,1,1,1]"のような文字列で与えられることを期待している
            lbws = [json.loads(lbw) for lbw in lbws]
        except Exception:
            raise ValueError(f"format of lbws are must be json / 層別適用率はJSON形式で書いてください")
        assert all(isinstance(lbw, list) for lbw in lbws), f"lbws are must be list / 層別適用率はリストにしてください"
        assert len(set(len(lbw) for lbw in lbws)) == 1, "all lbws should have the same length  / 層別適用率は同じ長さにしてください"
        assert all(len(lbw) in ACCEPTABLE for lbw in lbws), f"length of lbw are must be in {ACCEPTABLE} / 層別適用率の長さは{ACCEPTABLE}のいずれかにしてください"
        assert all(all(isinstance(weight, (int, float)) for weight in lbw) for lbw in lbws), f"values of lbs are must be numbers / 層別適用率の値はすべて数値にしてください"

        layer_num = len(lbws[0])
        is_sdxl = True if layer_num in SDXL_LAYER_NUM else False
        FLAGS = {
            "12": LAYER12.values(),
            "17": LAYER17.values(),
            "20": LAYER20.values(),
            "26": LAYER26.values(),
        }[str(layer_num)]
        LBW_TARGET_IDX = [i for i, flag in enumerate(FLAGS) if flag]

    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lora_metadata is not None:
            if v2 is None:
                v2 = lora_metadata.get(train_util.SS_METADATA_KEY_V2, None)  # returns string, SDXLはv2がないのでFalseのはず
            if base_model is None:
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        if lbw:
            lbw_weights = [1] * 26
            for index, value in zip(LBW_TARGET_IDX, lbw):
                lbw_weights[index] = value
            print(dict(zip(LAYER26.keys(), lbw_weights)))

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        logger.info(f"merging...")
        for key in tqdm(lora_sd.keys()):
            if "alpha" in key:
                continue

            if "lora_up" in key and concat:
                concat_dim = 1
            elif "lora_down" in key and concat:
                concat_dim = 0
            else:
                concat_dim = None

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            scale = abs(scale) if "lora_up" in key else scale # マイナスの重みに対応する。

            if lbw:
                index = get_lbw_block_index(key, is_sdxl)
                is_lbw_target = index in LBW_TARGET_IDX
                if is_lbw_target:
                    scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)
        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            dim = merged_sd[key_down].shape[0]
            perm = torch.randperm(dim)
            merged_sd[key_down] = merged_sd[key_down][perm]
            merged_sd[key_up] = merged_sd[key_up][:, perm]

    logger.info("merged model")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(v2, base_model, "networks.lora", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    assert (len(args.models) == len(args.ratios)
    ), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"
    if args.lbws:
        assert (len(args.models) == len(args.lbws)
        ), f"number of models must be equal to number of ratios / モデルの数と層別適用率の数は合わせてください"
    else:
        args.lbws = []  # zip_longestで扱えるようにlbws未使用時には空のリストにしておく

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    if args.sd_model is not None:
        logger.info(f"loading SD model: {args.sd_model}")

        (
            text_model1,
            text_model2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.sd_model, "cpu")

        merge_to_sd_model(text_model1, text_model2, unet, args.models, args.ratios, args.lbws, merge_dtype)

        if args.no_metadata:
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.sd_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, True, False, False, time.time(), title=title, merged_from=merged_from
            )

        logger.info(f"saving SD model to: {args.save_to}")
        sdxl_model_util.save_stable_diffusion_checkpoint(
            args.save_to, text_model1, text_model2, unet, 0, 0, ckpt_info, vae, logit_scale, sai_metadata, save_dtype
        )
    else:
        state_dict, metadata = merge_lora_models(args.models, args.ratios, args.lbws, merge_dtype, args.concat, args.shuffle)

        logger.info(f"calculating hashes and creating metadata...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                state_dict, False, False, True, True, False, time.time(), title=title, merged_from=merged_from
            )
            metadata.update(sai_metadata)

        logger.info(f"saving model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help="Stable Diffusion model to load: ckpt or safetensors file, merge LoRA models if omitted / 読み込むモデル、ckptまたはsafetensors。省略時はLoRAモデル同士をマージする",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument("--lbws", type=str, nargs="*", help="lbw for each model / それぞれのLoRAモデルの層別適用率")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims) / "
        + "マージの代わりに結合する（LoRAのdim(rank)は入力dimの合計になる）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle lora weight./ " + "LoRAの重みをシャッフルする",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
