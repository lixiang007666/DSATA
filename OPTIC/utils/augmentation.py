import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
from scipy.special import comb
from typing import Optional, List, Tuple, Union


def get_tta_transforms(image_size=512, gaussian_std=0.005, soft=False):
    n_pixels = image_size

    tta_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.2 if soft else 0.4,
            contrast=0.15 if soft else 0.3,
            saturation=0.25 if soft else 0.5,
            hue=0.03 if soft else 0.06
        ),
        transforms.Pad(padding=int(n_pixels / 8), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * gaussian_std),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    return tta_transforms

class RandomRotate(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = random.randrange(0, 3)
            if factor == 0:
                return x.flip(-1).transpose(-2, -1), factor    # 90
            elif factor == 1:
                return x.flip(-1).flip(-2), factor            # 180
            elif factor == 2:
                return x.transpose(-2, -1).flip(-1), factor    # 270
        else:
            return x, None

    def inverse(self, pred, factor):
        if factor is not None:
            if factor == 0:
                return pred.transpose(-2, -1).flip(-1)
            elif factor == 1:
                return pred.flip(-1).flip(-2)
            elif factor == 2:
                return pred.flip(-1).transpose(-2, -1)
        return pred


class RandomFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = random.randrange(0, 2)
            if factor == 0:
                return x.flip(-1), factor    # horizontal
            elif factor == 1:
                return x.flip(-2), factor    # vertical
        else:
            return x, None

    def inverse(self, pred, factor):
        if factor is not None:
            if factor == 0:
                return pred.flip(-1)
            elif factor == 1:
                return pred.flip(-2)
        return pred


class RotateAndFlip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor):
        # Flip
        if factor == 0:
            return x.flip(-1)  # horizontal
        elif factor == 1:
            return x.flip(-2)  # vertical
        # Rotate
        elif factor == 2:
            return x.flip(-1).transpose(-2, -1)  # 90
        elif factor == 3:
            return x.flip(-1).flip(-2)           # 180
        elif factor == 4:
            return x.transpose(-2, -1).flip(-1)  # 270
        return x

    def inverse(self, pred, factor):
        # Flip
        if factor == 0:
            return pred.flip(-1)
        elif factor == 1:
            return pred.flip(-2)
        # Rotate
        elif factor == 2:
            return pred.transpose(-2, -1).flip(-1)
        elif factor == 3:
            return pred.flip(-1).flip(-2)
        elif factor == 4:
            return pred.flip(-1).transpose(-2, -1)
        return pred


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, _, h, w = a_src.shape
    b = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1

    a_src[:, :, h1:h2, w1:w2] = a_trg[:, :, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    if L == 0:
        return src_img

    # Get FFT of both source and target
    fft_src_np = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))

    # Extract amplitude and phase
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, _ = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # Mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # Mutated FFT of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # Get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def fourier_augmentation(data, fda_beta=0.15):
    this_fda_beta = round(np.random.random() * fda_beta, 2)
    lowf_batch = np.random.permutation(data)
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data


def augment_lowfreq(input_tensor, beta=0.01, target_lowfreq=None, t=1.0):
    batch, _, imgH, imgW = input_tensor.size()
    lowfreq_H, lowfreq_W = int(imgH * beta), int(imgW * beta)
    padding_H, padding_W = (imgH - lowfreq_H) // 2, (imgW - lowfreq_W) // 2

    fft = torch.fft.fft2(input_tensor.clone(), dim=(-2, -1))

    # Extract amplitude and phase
    amp_src, pha_src = torch.abs(fft), torch.angle(fft)
    amp_src = torch.fft.fftshift(amp_src)

    # Low-frequency region
    low_freq = amp_src[:, :, padding_H:padding_H+lowfreq_H, padding_W:padding_W+lowfreq_W]
    if target_lowfreq is None:
        return input_tensor, low_freq
    else:
        target_lowfreq = torch.cat((target_lowfreq, low_freq), dim=0)

    # Augment with Gaussian noise based on statistics
    aug_lowfreq = torch.normal(
        mean=target_lowfreq.mean(dim=0).repeat(batch, 1, 1, 1),
        std=target_lowfreq.std(dim=0).repeat(batch, 1, 1, 1) / t
    )
    amp_src[:, :, padding_H:padding_H+lowfreq_H, padding_W:padding_W+lowfreq_W] = aug_lowfreq

    # Recompose FFT
    amp_src = torch.fft.ifftshift(amp_src)
    real = torch.cos(pha_src) * amp_src
    imag = torch.sin(pha_src) * amp_src
    fft_src_ = torch.complex(real=real, imag=imag)
    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
    return src_in_trg, target_lowfreq


# =============================================================================
# Location-Scale Augmentation (Bezier Curve, from GraTa)
# =============================================================================

class LocationScaleAugmentation:
    def __init__(self, vrange=(0., 1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints = nPoints
        self.nTimes = nTimes
        self.vrange = vrange
        self.background_threshold = background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array(
            [bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]
        ).astype(np.float32)

    def get_bezier_curve(self, points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints - 2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random() <= inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(
            location,
            self.vrange[0] - np.percentile(inputs, slide_limit),
            self.vrange[1] - np.percentile(inputs, 100 - slide_limit)
        )
        return np.clip(inputs * scale + location, self.vrange[0], self.vrange[1])

    def global_augmentation(self, image):
        image = self.non_linear_transformation(image, inverse=False)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image

    def local_augmentation(self, image, mask):
        output_image = np.zeros_like(image)
        mask = mask.astype(np.int32)
        reps = tuple((np.array(image.shape) // np.array(mask.shape)).tolist())
        mask = np.tile(mask, reps)

        output_image[mask == 0] = self.location_scale_transformation(
            self.non_linear_transformation(image[mask == 0], inverse=True, inverse_prop=1)
        )

        for c in range(1, np.max(mask) + 1):
            if (mask == c).sum() == 0:
                continue
            output_image[mask == c] = self.location_scale_transformation(
                self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5)
            )

        if self.background_threshold >= self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image

class StyleAugmentation:
    def __init__(self, strength='strong'):
        self.strength = strength
        if strength == 'strong':
            self.brightness_range = (0.5, 1.5)
            self.contrast_range = (0.5, 1.5)
            self.gamma_range = (0.5, 2.0)
            self.noise_std = 0.05
            self.blur_sigma_range = (0.5, 1.5)
            self.p_augment = 0.75
        else:  # weak
            self.brightness_range = (0.75, 1.25)
            self.contrast_range = (0.75, 1.25)
            self.gamma_range = (0.8, 1.2)
            self.noise_std = 0.02
            self.blur_sigma_range = (0.1, 0.5)
            self.p_augment = 0.25

    def brightness_transform(self, data):
        if random.random() < self.p_augment:
            factor = random.uniform(*self.brightness_range)
            data = data * factor
        return data

    def contrast_transform(self, data):
        if random.random() < self.p_augment:
            factor = random.uniform(*self.contrast_range)
            mean = data.mean()
            data = (data - mean) * factor + mean
        return data

    def gamma_transform(self, data):
        if random.random() < self.p_augment:
            gamma = random.uniform(*self.gamma_range)
            data_min = data.min()
            data_range = data.max() - data_min
            if data_range > 0:
                data = ((data - data_min) / data_range) ** gamma * data_range + data_min
        return data

    def gaussian_noise_transform(self, data):
        if random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, data.shape).astype(np.float32)
            data = data + noise
        return data

    def gaussian_blur_transform(self, data):
        if random.random() < 0.5:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(*self.blur_sigma_range)
            # Apply blur per channel
            if data.ndim == 4:  # (B, C, H, W)
                for b in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        data[b, c] = gaussian_filter(data[b, c], sigma=sigma)
            elif data.ndim == 3:  # (C, H, W)
                for c in range(data.shape[0]):
                    data[c] = gaussian_filter(data[c], sigma=sigma)
        return data

    def __call__(self, data):
        data = self.brightness_transform(data)
        data = self.contrast_transform(data)
        data = self.gamma_transform(data)
        data = self.gaussian_noise_transform(data)
        if self.strength == 'strong':
            data = self.gaussian_blur_transform(data)
        return np.clip(data, 0, 1).astype(np.float32)


# =============================================================================
# Unified Augmentor Class
# =============================================================================

class TTAAugmentor:
    AUGMENTATION_TYPES = ['tta', 'spatial', 'fourier', 'style_strong', 'style_weak', 'bezier', 'combined']
    
    def __init__(
        self, 
        image_size: int = 512, 
        gaussian_std: float = 0.005, 
        soft: bool = False, 
        device: str = 'cuda',
        aug_type: str = 'tta',
        fda_beta: float = 0.15,
        combined_augs: Optional[List[str]] = None
    ):

        self.image_size = image_size
        self.gaussian_std = gaussian_std
        self.device = device
        self.aug_type = aug_type
        self.fda_beta = fda_beta
        self.combined_augs = combined_augs or ['spatial', 'style_strong']
        
        # Initialize transforms based on type
        self.tta_transform = get_tta_transforms(image_size, gaussian_std, soft)
        self.rotate_and_flip = RotateAndFlip()
        self.random_rotate = RandomRotate(p=0.5)
        self.random_flip = RandomFlip(p=0.5)
        self.style_strong = StyleAugmentation(strength='strong')
        self.style_weak = StyleAugmentation(strength='weak')
        self.bezier_aug = LocationScaleAugmentation()
    
    def apply_tta_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        augmented_batch = []
        for i in range(x.shape[0]):
            img = x[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            augmented_img = self.tta_transform(img)
            if isinstance(augmented_img, torch.Tensor):
                augmented_batch.append(augmented_img)
            else:
                augmented_batch.append(transforms.ToTensor()(augmented_img))
        return torch.stack(augmented_batch).to(self.device)
    
    def apply_spatial_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        x_aug, _ = self.random_rotate(x)
        x_aug, _ = self.random_flip(x_aug)
        return x_aug
    
    def apply_fourier_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        x_aug = fourier_augmentation(x_np, fda_beta=self.fda_beta)
        return torch.from_numpy(x_aug).to(dtype=torch.float32, device=self.device)
    
    def apply_style_augmentation(self, x: torch.Tensor, strength: str = 'strong') -> torch.Tensor:
        x_np = x.cpu().numpy()
        if strength == 'strong':
            x_aug = self.style_strong(x_np)
        else:
            x_aug = self.style_weak(x_np)
        return torch.from_numpy(x_aug).to(dtype=torch.float32, device=self.device)
    
    def apply_bezier_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        x_aug = self.bezier_aug.global_augmentation(x_np)
        return torch.from_numpy(x_aug).to(dtype=torch.float32, device=self.device)
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:

        if self.aug_type == 'tta':
            return self.apply_tta_augmentation(x)
        elif self.aug_type == 'spatial':
            return self.apply_spatial_augmentation(x)
        elif self.aug_type == 'fourier':
            return self.apply_fourier_augmentation(x)
        elif self.aug_type == 'style_strong':
            return self.apply_style_augmentation(x, strength='strong')
        elif self.aug_type == 'style_weak':
            return self.apply_style_augmentation(x, strength='weak')
        elif self.aug_type == 'bezier':
            return self.apply_bezier_augmentation(x)
        elif self.aug_type == 'combined':
            return self.apply_combined_augmentation(x)
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}. "
                           f"Available types: {self.AUGMENTATION_TYPES}")
    
    def apply_combined_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = x
        for aug_name in self.combined_augs:
            if aug_name == 'tta':
                x_aug = self.apply_tta_augmentation(x_aug)
            elif aug_name == 'spatial':
                x_aug = self.apply_spatial_augmentation(x_aug)
            elif aug_name == 'fourier':
                x_aug = self.apply_fourier_augmentation(x_aug)
            elif aug_name == 'style_strong':
                x_aug = self.apply_style_augmentation(x_aug, strength='strong')
            elif aug_name == 'style_weak':
                x_aug = self.apply_style_augmentation(x_aug, strength='weak')
            elif aug_name == 'bezier':
                x_aug = self.apply_bezier_augmentation(x_aug)
        return x_aug
    
    def get_consistency_augmentations(self, x: torch.Tensor, num_augs: int = 5) -> List[torch.Tensor]:
        augmented_list = []
        for factor in range(num_augs):
            x_aug = self.rotate_and_flip(x, factor)
            augmented_list.append(x_aug)
        return augmented_list
    
    def inverse_consistency_augmentation(self, pred: torch.Tensor, factor: int) -> torch.Tensor:
        return self.rotate_and_flip.inverse(pred, factor)
