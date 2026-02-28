"""Test CNNEncoder and PixelActorCritic shapes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from networks import CNNEncoder, PixelActorCritic


def test_encoder():
    print("=" * 50)
    print("测试 CNNEncoder")
    print("=" * 50)

    encoder = CNNEncoder(in_channels=9, latent_dim=256)

    fake_images = torch.randn(4, 9, 84, 84)
    print(f"输入形状:  {fake_images.shape}")

    latent = encoder(fake_images)
    print(f"输出形状:  {latent.shape}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"CNN 参数量: {total_params:,}")
    print("CNNEncoder OK\n")


def test_pixel_actor_critic():
    print("=" * 50)
    print("测试 PixelActorCritic")
    print("=" * 50)

    network = PixelActorCritic(in_channels=9, act_dim=8)

    # 单个观测 (select_action 时)
    single_obs = torch.randn(9, 84, 84)
    with torch.no_grad():
        action, log_prob, entropy, value = network.get_action_and_value(single_obs)
    print(f"[单个观测]")
    print(f"  obs:      {single_obs.shape}")
    print(f"  action:   {action.shape}")
    print(f"  log_prob: {log_prob.shape}")
    print(f"  value:    {value.shape}")

    # batch 观测 (update 时)
    batch_obs = torch.randn(32, 9, 84, 84)
    fake_actions = torch.randn(32, 8)

    action, log_prob, entropy, value = network.get_action_and_value(
        batch_obs, fake_actions
    )
    print(f"\n[Batch, batch_size=32]")
    print(f"  obs:      {batch_obs.shape}")
    print(f"  action:   {action.shape}")
    print(f"  log_prob: {log_prob.shape}")
    print(f"  value:    {value.shape}")

    # get_value (bootstrap)
    with torch.no_grad():
        val = network.get_value(single_obs)
    print(f"\n[get_value]")
    print(f"  value: {val.shape}, value = {val.item():.4f}")

    total = sum(p.numel() for p in network.parameters())
    encoder_params = sum(p.numel() for p in network.encoder.parameters())
    head_params = total - encoder_params
    print(f"\n总参数量:    {total:,}")
    print(f"  CNN 编码器: {encoder_params:,} ({encoder_params/total*100:.1f}%)")
    print(f"  Policy 头:  {head_params:,} ({head_params/total*100:.1f}%)")
    print("\nPixelActorCritic OK")


if __name__ == "__main__":
    test_encoder()
    test_pixel_actor_critic()
