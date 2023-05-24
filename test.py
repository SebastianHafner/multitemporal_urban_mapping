import torch
import torch.nn.functional as F
from torchvision import transforms

def example():
    x = torch.randn(1, 172, 220, 156)
    kc, kh, kw = 32, 32, 32  # kernel size
    dc, dh, dw = 32, 32, 32  # stride
    # Pad to multiples of 32
    x = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
                  x.size(1) % kh // 2, x.size(1) % kh // 2,
                  x.size(0) % kc // 2, x.size(0) % kc // 2))

    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, kc, kh, kw)
    print(patches.shape)

    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    # Check for equality
    print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())


def mine():
    # https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
    patch = (3, 3)
    x = torch.arange(16).float()
    print(x, x.shape)
    x2d = x.reshape(1, 1, 4, 4)
    print(x2d, x2d.shape)
    x2d = transforms.Pad((1, 1, 1, 1), padding_mode='edge')(x2d)
    print(x2d, x2d.shape)
    h, w = patch
    c = x2d.size(1)
    print(c)  # channels
    # unfold(dimension, size, step)
    r = x2d.unfold(2, h, 1).unfold(3, w, 1)
    # print(r, r.shape)  # (1, 1, 2, 2, 3, 3)
    print(r[:, :, 1, 1, :, :])
    # r = r.transpose(1, 3).reshape(-1, c, h, w)
    # print(r.shape)
    # print(r)  # result


if __name__ == '__main__':
    mine()
