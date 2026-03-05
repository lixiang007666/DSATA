import torch
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import kornia.augmentation as K
import open_clip
from types import SimpleNamespace


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, aug_prob, p_min, p_max, patch=False):
        super().__init__()
        self.output_size = output_size
        self.aug_prob = aug_prob
        self.patch = patch
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=aug_prob, padding_mode="border"),
            K.RandomPerspective(0.7, p=aug_prob),
        )
        self.random_patch = K.RandomResizedCrop(size=(128,128), scale=(p_min,p_max))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input, num_patch=32, is_global=False):
        if self.patch:
            if is_global:
                input = input.repeat(num_patch,1,1,1)
            else:
                input_patches = []
                for i in range(num_patch):
                    if self.aug_prob > 0.0:
                        tmp = self.augmentations(self.random_patch(input))
                    else:
                        tmp = self.random_patch(input)
                    input_patches.append(tmp)
                input = torch.cat(input_patches,dim=0)
        else:
            input_patches = []
            for i in range(num_patch):
                input_patches.append(self.augmentations(input))
            input = torch.cat(input_patches,dim=0)
        
        resized_images = self.avg_pool(input)
        return resized_images


def cosine_distance(x, y, use_cosine=True):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    return distance


# Semantic Anchor Regularization (SAR) Encoder
class SAREncoder:
    def __init__(self, args) -> None:
        self.vision_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.args = args
        self.device = args.device
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.vision_model.to(self.device).eval()

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.image_augmentations = ImageAugmentations(224, aug_prob=1, p_min=0.01, p_max=0.3, patch=False)

    def compute_sar_loss(self, x_in, text):
        text_inputs = self.tokenizer([text]).to(self.device)
        text_embed = self.vision_model.encode_text(text_inputs).float()
        sar_loss = torch.tensor(0)
        augmented_input = self.image_augmentations(x_in, num_patch=self.args.n_patch).add(1).div(2)
        normalized_input = self.normalize(augmented_input).to(self.device)
        image_embeds = self.vision_model.encode_image(normalized_input).float()
        dists = cosine_distance(image_embeds, text_embed)
        for i in range(self.args.batch_size):
            sar_loss = sar_loss + dists[i :: self.args.batch_size].mean()
        return sar_loss


def main():
    args = SimpleNamespace(
        aug_prob=0.8,      
        p_min=0.01,         
        p_max=0.3,          
        n_patch=32,         
        batch_size=1      
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SAREncoder(args)
    encoder.device = device
    x_in = torch.randn(args.batch_size, 3, 352, 352).to(device) 
    text = "a photo of a polyp in colonoscopy"
    sar_loss = encoder.compute_sar_loss(x_in, text)
    print("SAR Loss:", sar_loss.item())


if __name__ == "__main__":
    main()
