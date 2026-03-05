import torch
import torch.nn as nn
import torch.nn.functional as F


# Frequency-domain Convolutional Prompt (FCP)
class FCP(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=512, hidden_channels=32, init_radius_ratio=0.1):
        super().__init__()
        self.prompt_alpha = prompt_alpha
        self.image_size = image_size
        
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size) // 2
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1, bias=True),
        )
        
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.radius_ratio = nn.Parameter(torch.tensor(init_radius_ratio), requires_grad=True)
        self.smoothness = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.register_buffer('distance_map', self._create_distance_map())
        self._init_weights()
        self.register_buffer('data_prompt', torch.zeros((1, 3, self.prompt_size, self.prompt_size)))
    
    def _create_distance_map(self):
        H, W = self.image_size, self.image_size
        center_h, center_w = H / 2, W / 2
        y_coords = torch.arange(H).float() - center_h
        x_coords = torch.arange(W).float() - center_w
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt(xx**2 + yy**2)
        max_distance = torch.sqrt(torch.tensor(center_h**2 + center_w**2))
        distance_normalized = distance / max_distance
        return distance_normalized.unsqueeze(0).unsqueeze(0)
    
    def _init_weights(self):
        last_conv = self.conv_net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
        for m in self.conv_net[:-1]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def generate_soft_mask(self):
        radius_ratio_clamped = torch.clamp(self.radius_ratio, 0.01, 0.5)
        smoothness_clamped = torch.clamp(self.smoothness, 1.0, 100.0)
        mask = torch.sigmoid((radius_ratio_clamped - self.distance_map) * smoothness_clamped)
        return mask

    def update(self, init_data):
        pass

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        B, C, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        soft_mask = self.generate_soft_mask()
        prompt_residual = self.conv_net(amp_src)
        prompt_residual = self.scale * prompt_residual
        amp_src_ = amp_src + prompt_residual * soft_mask
        
        amp_low = amp_src[:, :, 
                          self.padding_size:self.padding_size + self.prompt_size, 
                          self.padding_size:self.padding_size + self.prompt_size]
        
        self.data_prompt = prompt_residual[:, :,
                                           self.padding_size:self.padding_size + self.prompt_size,
                                           self.padding_size:self.padding_size + self.prompt_size].detach()

        amp_src_ = torch.fft.ifftshift(amp_src_)
        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        
        return src_in_trg, amp_low
    
    def get_mask_info(self):
        return {
            'radius_ratio': self.radius_ratio.item(),
            'smoothness': self.smoothness.item(),
            'effective_radius_pixels': self.radius_ratio.item() * self.image_size / 2
        }
