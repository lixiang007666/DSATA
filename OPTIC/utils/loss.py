import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()


@torch.jit.script
def softmax_entropy_sample(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def _consistency_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()


def cps_loss(logits: torch.Tensor, high_margin: float) -> torch.Tensor:
    entropys = softmax_entropy_sample(logits)
    filter_ids = torch.where(entropys < high_margin)
    entropys = entropys[filter_ids]
    
    if entropys.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    coeff = 1 / (torch.exp(entropys.clone().detach() - high_margin))
    weighted_entropy = entropys.mul(coeff).mean(0)
    
    return weighted_entropy


def con_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
             scale: float = 0.01) -> torch.Tensor:
    num_classes = student_logits.shape[1]
    return scale * num_classes * _consistency_loss(student_logits, teacher_logits.detach())


class SARLoss(nn.Module):
    def __init__(self, sar_encoder, text_prompt: str, lambda_sar: float = 0.01):
        super().__init__()
        self.sar_encoder = sar_encoder
        self.text_prompt = text_prompt
        self.lambda_sar = lambda_sar
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sar_loss = self.sar_encoder.compute_sar_loss(images, self.text_prompt)
        return self.lambda_sar * sar_loss


class DSATALoss(nn.Module):
    def __init__(self, sar_encoder=None, sar_text: str = "", 
                 lambda_sar: float = 0.01, high_margin: Optional[float] = None):
        super().__init__()
        self.high_margin = high_margin if high_margin is not None else torch.log(torch.tensor(2.0)).item() * 0.20
        self.lambda_sar = lambda_sar
        self.sar_text = sar_text
        self.sar_encoder = sar_encoder
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                prompt_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cps = cps_loss(student_logits, self.high_margin)
        con = con_loss(student_logits, teacher_logits)
        total_loss = cps + con
        
        losses = {
            'cps': cps,
            'con': con,
            'total': total_loss
        }
        
        if self.sar_encoder is not None and prompt_images is not None:
            sar = self.sar_encoder.compute_sar_loss(prompt_images, self.sar_text)
            losses['sar'] = sar
            losses['total'] = total_loss + self.lambda_sar * sar
        
        return losses
