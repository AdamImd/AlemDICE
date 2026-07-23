#!/usr/bin/env python3
"""Render a full-world MP4 after an episode from saved EnvState snapshots."""

from __future__ import annotations

import argparse
import gzip
import pickle
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from alem.alem_coop.constants import (
    BLOCK_PIXEL_SIZE_IMG,
    TEXTURES,
    ItemType,
    load_player_specific_textures,
)

HEADER_HEIGHT = 64
AGENT_BADGE_COLORS = (
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (234, 179, 8),
    (168, 85, 247),
    (236, 72, 153),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path, help="Saved *_states.pkl.gz bundle")
    parser.add_argument("output", type=Path, help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth saved state")
    parser.add_argument(
        "--title", default="Full-world replay", help="Title drawn above the world map"
    )
    return parser


def _as_rgb(texture: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(texture)[..., :3], 0, 255).astype(np.uint8)


def _alpha_composite(
    canvas: np.ndarray, texture: np.ndarray, alpha: np.ndarray, row: int, col: int
) -> None:
    tile = canvas[row : row + texture.shape[0], col : col + texture.shape[1]]
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.ndim == 2:
        alpha_array = alpha_array[..., None]
    elif alpha_array.shape[-1] != 1:
        alpha_array = alpha_array[..., :1]
    if alpha_array.max(initial=0) > 1:
        alpha_array = alpha_array / 255.0
    rgb = np.asarray(texture, dtype=np.float32)[..., :3]
    tile[:] = np.clip(tile * (1.0 - alpha_array) + rgb * alpha_array, 0, 255).astype(np.uint8)


def _overlay_rgba(canvas: np.ndarray, rgba: np.ndarray, row: int, col: int) -> None:
    rgba = np.asarray(rgba)
    _alpha_composite(canvas, rgba[..., :3], rgba[..., 3], row, col)


def _overlay_mobs(canvas, state, level, tile_size, textures) -> None:
    mob_specs = (
        (state.melee_mobs, "melee_mob_textures", "melee_mob_texture_alphas"),
        (state.passive_mobs, "passive_mob_textures", "passive_mob_texture_alphas"),
        (state.ranged_mobs, "ranged_mob_textures", "ranged_mob_texture_alphas"),
    )
    for mobs, texture_key, alpha_key in mob_specs:
        positions = np.asarray(mobs.position[level])
        masks = np.asarray(mobs.mask[level], dtype=bool)
        type_ids = np.asarray(mobs.type_id[level], dtype=int)
        mob_textures = np.asarray(textures[texture_key])
        mob_alphas = np.asarray(textures[alpha_key])
        for position, mask, type_id in zip(positions, masks, type_ids, strict=True):
            if not mask:
                continue
            row, col = (int(position[0]) * tile_size, int(position[1]) * tile_size)
            _alpha_composite(canvas, mob_textures[type_id], mob_alphas[type_id], row, col)


def _overlay_projectiles(canvas, state, level, tile_size, textures) -> None:
    projectile_textures = np.asarray(textures["projectile_textures"])
    projectile_alphas = np.asarray(textures["projectile_texture_alphas"])
    specs = (
        (state.mob_projectiles, state.mob_projectile_directions),
        (state.player_projectiles, state.player_projectile_directions),
    )
    for projectiles, directions in specs:
        positions = np.asarray(projectiles.position[level])
        masks = np.asarray(projectiles.mask[level], dtype=bool)
        type_ids = np.asarray(projectiles.type_id[level], dtype=int)
        level_directions = np.asarray(directions[level])
        for position, mask, type_id, direction in zip(
            positions, masks, type_ids, level_directions, strict=True
        ):
            if not mask:
                continue
            texture = projectile_textures[type_id]
            alpha = projectile_alphas[type_id]
            if direction[0] > 0 or direction[1] > 0:
                texture = np.flip(texture, axis=0)
                alpha = np.flip(alpha, axis=0)
            if direction[1] != 0:
                texture = np.transpose(texture, (1, 0, 2))
                alpha = np.transpose(alpha, (1, 0, 2))
            row, col = (int(position[0]) * tile_size, int(position[1]) * tile_size)
            _alpha_composite(canvas, texture, alpha, row, col)


def _render_state(
    state, static_params, textures, player_textures, step, total_steps, title
) -> np.ndarray:
    tile_size = BLOCK_PIXEL_SIZE_IMG
    level = int(np.asarray(state.player_level))
    world_map = np.asarray(state.map[level], dtype=int)
    block_textures = np.asarray(textures["block_textures"])[..., :3]
    tiles = block_textures[world_map]
    world = np.transpose(tiles, (0, 2, 1, 3, 4)).reshape(
        world_map.shape[0] * tile_size, world_map.shape[1] * tile_size, 3
    )
    world = np.clip(world, 0, 255).astype(np.uint8)

    item_map = np.asarray(state.item_map[level], dtype=int)
    item_textures = np.asarray(textures["full_map_item_textures"])
    for item_type in range(1, len(ItemType)):
        for row, col in np.argwhere(item_map == item_type):
            rgba = item_textures[item_type, :tile_size, :tile_size]
            _overlay_rgba(world, rgba, int(row) * tile_size, int(col) * tile_size)

    _overlay_mobs(world, state, level, tile_size, textures)
    _overlay_projectiles(world, state, level, tile_size, textures)

    if level > 0:
        light = np.asarray(state.light_map[level], dtype=np.float32)
        light = np.repeat(np.repeat(light, tile_size, axis=0), tile_size, axis=1)
        world = np.clip(world.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)
    else:
        daylight = float(np.asarray(state.light_level))
        world = np.clip(world.astype(np.float32) * (0.35 + 0.65 * daylight), 0, 255).astype(
            np.uint8
        )

    positions = np.asarray(state.player_position, dtype=int)
    directions = np.asarray(state.player_direction, dtype=int)
    alive = np.asarray(state.player_alive, dtype=bool)
    sleeping = np.asarray(state.is_sleeping, dtype=bool)
    for agent_id in range(static_params.player_count):
        texture_id = 4 if sleeping[agent_id] else max(directions[agent_id] - 1, 0)
        if not alive[agent_id]:
            texture_id = 5
        rgba = np.asarray(player_textures.player_textures[agent_id, texture_id])
        row = int(positions[agent_id, 0]) * tile_size
        col = int(positions[agent_id, 1]) * tile_size
        _overlay_rgba(world, rgba, row, col)

    frame = Image.new("RGB", (world.shape[1], world.shape[0] + HEADER_HEIGHT), (15, 23, 42))
    frame.paste(Image.fromarray(world), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=14)
    title_font = ImageFont.load_default(size=18)
    draw.text((12, 8), title, fill=(248, 250, 252), font=title_font)
    draw.text(
        (12, 36),
        f"Step {step + 1:0{len(str(total_steps))}d} / {total_steps}     Level {level}",
        fill=(203, 213, 225),
        font=font,
    )
    legend_x = frame.width - static_params.player_count * 68 - 10
    for agent_id in range(static_params.player_count):
        color = AGENT_BADGE_COLORS[agent_id % len(AGENT_BADGE_COLORS)]
        x = legend_x + agent_id * 68
        draw.ellipse((x, 21, x + 18, 39), fill=color, outline=(248, 250, 252), width=1)
        draw.text((x + 5, 22), str(agent_id), fill=(255, 255, 255), font=font)
        draw.text((x + 23, 23), f"A{agent_id}", fill=(226, 232, 240), font=font)

        badge_row = HEADER_HEIGHT + int(positions[agent_id, 0]) * tile_size
        badge_col = int(positions[agent_id, 1]) * tile_size
        draw.ellipse(
            (badge_col, badge_row, badge_col + 11, badge_row + 11),
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )
        draw.text((badge_col + 3, badge_row - 1), str(agent_id), fill=(255, 255, 255))
    return np.asarray(frame)


def render(states_path: Path, output_path: Path, fps: float, stride: int, title: str) -> None:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    if fps <= 0 or stride < 1:
        raise ValueError("--fps must be positive and --stride must be at least 1")
    with gzip.open(states_path, "rb") as handle:
        payload = pickle.load(handle)
    states = payload["states"]
    static_params = payload["static_env_params"]
    if not states:
        raise ValueError("State bundle contains no frames")

    textures = TEXTURES[BLOCK_PIXEL_SIZE_IMG]
    player_textures = load_player_specific_textures(textures, static_params.player_count)
    first = _render_state(
        states[0], static_params, textures, player_textures, 0, len(states), title
    )
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for step in range(0, len(states), stride):
            frame = (
                first
                if step == 0
                else _render_state(
                    states[step],
                    static_params,
                    textures,
                    player_textures,
                    step,
                    len(states),
                    title,
                )
            )
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the replay")


def main() -> None:
    args = _parser().parse_args()
    render(args.states.resolve(), args.output.resolve(), args.fps, args.stride, args.title)


if __name__ == "__main__":
    main()
