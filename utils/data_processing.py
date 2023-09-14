def generate_patch(patch_size, image):
    num_channels = image.size(0)  # Dataloader거친 후 Batch_size, Channel, W, H
    patches_w = image.unfold(
        1, patch_size, patch_size
    )  # dimension, size, step, w기준으로 patch나누는과정
    patches = patches_w.unfold(2, patch_size, patch_size)  # h기준으로 patch화
    patches = patches.reshape(num_channels, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # patch들,, channel, w,h
    flatten_patches = patches.reshape(patches.size(0), -1)  # flatten과정
    return flatten_patches
