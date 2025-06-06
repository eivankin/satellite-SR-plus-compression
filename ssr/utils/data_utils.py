import torch

def has_black_pixels(tensor):
    # Sum along the channel dimension to get a 2D tensor [height, width]
    channel_sum = torch.sum(tensor, dim=0)

    # Check if any pixel has a sum of 0, indicating black
    black_pixels = (channel_sum.view(-1) == 0).any()

    return black_pixels


def get_random_nonzero_extent(tensor, scale=4):
    """
    Finds a random 128x128 crop in a [C, W, H] tensor with no black (all-zero) pixels.
    Returns the (y1, y2, x1, x2) indices of the crop.
    Raises ValueError if no such crop is found after 100 attempts.
    """
    C, W, H = tensor.shape
    crop_size = 128
    max_attempts = 100
    for i in range(max_attempts):
        x1 = torch.randint(scale, W - crop_size + 1 - scale, (1,)).item()
        y1 = torch.randint(scale, H - crop_size + 1 - scale, (1,)).item()
        crop = tensor[:, x1:x1+crop_size, y1:y1+crop_size]
        if x1 % scale == 0 and y1 % scale == 0 and (not has_black_pixels(crop) or i > 0.75 * max_attempts):
            return (x1, x1+crop_size, y1, y1+crop_size)
    return (x1, x1 + crop_size, y1, y1 + crop_size)
    # raise ValueError(f"No non-black 128x128 crop found after {max_attempts} attempts")