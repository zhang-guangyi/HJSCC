import torch.nn as nn


class config:

    train_data_dir = ['/path/to/train/dataset']
    test_data_dir = ['/path/to/test/dataset']

    batch_size = 8
    num_workers = 8

    print_step = 50
    plot_step = 1000
    logger = None

    # training details
    image_dims = (3, 256, 256)
    lr = 1e-4
    aux_lr = 1e-3
    distortion_metric = 'MSE'  # 'MS-SSIM'

    #conv details
    num_iteration=3
    dec_num_layer=3
    num_iter_ft=16
    dec_num_unit=16
    dec_kernel_size=9


    use_side_info = False
    test = 0
    train_lambda = 64
    eta = 0.4

    channel = {"type": 'awgn', 'chan_param': 10}
    multiple_rate = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    rate_choice_1 = range(2, 33, 2)
    rate_choice_2 = range(2, 33, 2)
    rate_choice_3 = range(4, 97, 4)
    rate_choice_4 = range(1, 9, 1)


    fe_kwargs_0 = dict(
        input_resolution=(256 // 64, 256// 64),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_1
    )
    fe_kwargs_1 = dict(
        input_resolution=(256 // 32, 256// 32),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_2
    )
    fe_kwargs_2 = dict(
        input_resolution=(256 // 16, 256// 16),
        embed_dim=256, depths=[4], num_heads=[4],
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_3
    )
    fe_kwargs_3 = dict(
        input_resolution=(256 // 8, 256// 8),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=32, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_4
    )

    fd_kwargs_0 = dict(
        input_resolution=(256 // 64, 256// 64),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_1
    )
    fd_kwargs_1 = dict(
        input_resolution=(256 // 32, 256// 32),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_2
    )
    fd_kwargs_2 = dict(
        input_resolution=(256 // 16, 256// 16),
        embed_dim=256, depths=[4], num_heads=[4],
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_3
    )
    fd_kwargs_3 = dict(
        input_resolution=(256 // 8, 256// 8),
        embed_dim=256, depths=[4], num_heads=[2],
        window_size=32, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=rate_choice_4
    )
