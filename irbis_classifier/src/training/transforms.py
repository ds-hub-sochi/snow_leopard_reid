import albumentations as A


val_transfroms: A.Compose = A.Compose(
    [
        A.LongestMaxSize(
            max_size=256,
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=256,
            min_width=256,
            position='center',
            border_mode=0,
            fill=0,
            p=1.0,
        ),
        A.Resize(
            height=224,
            width=224,
            p=1.0,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.transforms.ToTensorV2(p=1.0),  # pylint: disable=no-member
    ]
)

train_transforms: A.Compose = A.Compose(
    [
        A.LongestMaxSize(
            max_size=256,
            p=1.0,
        ),
        A.ToGray(
            num_output_channels=3,
            p=0.10,
        ),
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=(-20, 20),
                    sat_shift_limit=(-30, 30),
                    val_shift_limit=(-20, 20),
                    p=0.85,
                ),
                A.Equalize(
                    mode='cv',
                    by_channels=True,
                    p=0.85,
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.85,
                )
            ],
            p=1.0
        ),
        A.OneOf(
            [
                A.Defocus(
                    radius=(3, 5),
                    alias_blur=(0.01, 0.02),
                    p=0.85,
                ),
                A.GlassBlur(
                    sigma=0.5,
                    max_delta=2,
                    iterations=1,
                    mode='fast',
                    p=0.85,
                ),
                A.Blur(
                    blur_limit=(3, 7),
                    p=0.85,
                ),
                A.MotionBlur(
                    blur_limit=(3, 7),
                    angle_range=(0, 360),
                    direction_range=(-1.0, 1.0),
                    allow_shifted=True,
                    p=0.85,
                )
            ],
            p=1.0
        ),
        A.OneOf(
            [
                A.RandomSunFlare(
                    flare_roi=(0.0, 0.0, 1.0, 0.25),
                    num_flare_circles_range=(2, 5),
                    src_radius=256 // 2,
                    p=0.05,
                ),
                A.RandomRain(
                    p=0.05,
                ),
                A.SaltAndPepper(
                    amount=(0.05, 0.10),
                    p=0.10,
                ),
            ],
            p=1.0,
        ),
        A.SafeRotate(
            limit=(-30, 30),
            p=0.85,
        ),
        A.RandomCrop(
            height=224,
            width=224,
            pad_if_needed=True,
            border_mode=0,
            fill=0,
            pad_position='center',
            p=1.0,
        ),
        A.HorizontalFlip(
            p=0.85,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.transforms.ToTensorV2(p=1.0),  # pylint: disable=no-member
    ]
)