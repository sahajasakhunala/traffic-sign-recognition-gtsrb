from torchvision import transforms

def get_train_transforms(image_size: int) -> transforms.Compose:
    """Returns training image transformations including augmentations.
    
    Random horizontal/vertical flips are intentionally omitted as they 
    would create invalid signs (e.g. Keep Right -> Keep Left).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            shear=5,
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],  # Official GTSRB statistics
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])

def get_val_transforms(image_size: int) -> transforms.Compose:
    """Returns standard validation/testing image transformations."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])
