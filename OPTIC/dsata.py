import glob
import math
import os
import random
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.fcp import FCP
from utils.metrics import calculate_metrics
from utils.inject_fra import inject_trainable_fra
from utils.sar import SAREncoder
from utils.loss import DSATALoss
from types import SimpleNamespace
from networks.segformer import SegFormer
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from tqdm import tqdm
import pdb
import cv2
import torch.nn.functional as F
from copy import deepcopy
from utils.augmentation import TTAAugmentor
from types import SimpleNamespace

torch.set_num_threads(1)

def save_images(batch, names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img_tensor, name in zip(batch, names):
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        out_path = os.path.join(save_dir, os.path.basename(str(name)))
        cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def save_predictions(seg_output, names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for mask_np, name in zip(seg_output, names):
        combined_mask = np.zeros_like(mask_np, dtype=np.uint8)
        combined_mask[mask_np == 1] = 128
        combined_mask[mask_np == 2] = 255
        out_path = os.path.join(save_dir, os.path.basename(str(name)))
        cv2.imwrite(out_path, combined_mask)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    return ema_model


class DSATA:
    def __init__(self, config):
        config.seed_torch(42)
        self.weight_epoch = config.epoch
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, f'DSATA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        self.prompt_image_dir = os.path.join(log_root, f'promptimage_{time_now}')
        self.predict_dir = os.path.join(log_root, f'predict_{time_now}')
        os.makedirs(self.prompt_image_dir, exist_ok=True)
        os.makedirs(self.predict_dir, exist_ok=True)


        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid':
                target_test_csv.append(target + '_train.csv')
                target_test_csv.append(target + '_test.csv')
            else:
                target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters

        # SAR loss
        self.lambda_sar = config.lambda_sar
        self.sar_text = config.sar_text

        # Augmentation
        self.aug_type = config.aug_type

        # Teacher-Student parameters
        self.alpha_teacher = config.alpha_teacher
        self.lambda_consistency = config.lambda_consistency
        self.lambda_entropy = config.lambda_entropy

        # Initialize models
        self.build_model()

        # Data augmentation
        self.augmentor = TTAAugmentor(
            image_size=self.image_size, 
            device=self.device,
            aug_type=self.aug_type
        )
        self.high_margin = math.log(2) * 0.20

        # Initialize loss function
        self.criterion = DSATALoss(
            sar_encoder=self.sar_encoder,
            sar_text=self.sar_text,
            lambda_sar=self.lambda_sar,
            high_margin=self.high_margin
        )

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        # Student model (with prompt)
        self.prompt = FCP(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.student_model = SegFormer(num_classes=self.out_ch+1, phi='b5', pretrained=False, convert=False, newBN=AdaBN).to(self.device)

        # Teacher model (without prompt, EMA updated)
        self.teacher_model = SegFormer(num_classes=self.out_ch+1, phi='b5', pretrained=False, convert=False, newBN=AdaBN).to(self.device)

        # Load pre-trained weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(self.load_model, 'segf/last_epoch_weights.pth'), map_location=device)
        self.student_model.load_state_dict(checkpoint, strict=True)
        self.teacher_model.load_state_dict(checkpoint, strict=True)

        fra_params, fra_names = inject_trainable_fra(
            model=self.student_model, 
            target_replace_module=["CrossAttention", "Attention"], 
            r=1, r2=128
        )

        # Initialize teacher as copy of student (with FRA)
        self.teacher_model = deepcopy(self.student_model)
        for param in self.teacher_model.parameters():
            param.detach_()

        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Freeze student model, only train FRA adapters
        for name, param in self.student_model.named_parameters():
            if 'fra_' not in name:
                param.requires_grad = False

        # Collect trainable parameters
        prompt_params = list(self.prompt.parameters())
        fra_model_params = [p for n, p in self.student_model.named_parameters() if 'fra_' in n]

        if self.optim == 'SGD':
            param_groups = [
                {'params': prompt_params, 'lr': self.lr, 'momentum': self.momentum, 
                 'nesterov': True, 'weight_decay': self.weight_decay},
                {'params': fra_model_params, 'lr': self.lr * 1e-2, 'momentum': self.momentum,
                 'nesterov': True, 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.SGD(param_groups)
        elif self.optim == 'Adam':
            param_groups = [
                {'params': prompt_params, 'lr': self.lr, 'betas': self.betas, 
                 'weight_decay': self.weight_decay},
                {'params': fra_model_params, 'lr': self.lr * 1e-2, 'betas': self.betas,
                 'weight_decay': self.weight_decay}
            ]
            self.optimizer = torch.optim.Adam(param_groups)

        # Initialize SAR encoder for semantic alignment loss
        self.sar_encoder = SAREncoder(SimpleNamespace(
            aug_prob=0.8, p_min=0.01, p_max=0.3, n_patch=32, batch_size=1, device=self.device
        ))
        self.sar_encoder.device = self.device

    def print_prompt(self):
        prompt_params = 0
        for p in self.prompt.parameters():
            prompt_params += p.numel()

        fra_params = 0
        for name, param in self.student_model.named_parameters():
            if 'fra_' in name:
                fra_params += param.numel()

        total_params = prompt_params + fra_params

        print("Prompt parameters: {}".format(prompt_params))
        print("FRA parameters: {}".format(fra_params))
        print("Total trainable parameters: {}".format(total_params))
        print("Prompt scale init: {:.6f}".format(self.prompt.scale.item()))
        mask_info = self.prompt.get_mask_info()
        print("Mask info: radius_ratio={:.4f}, smoothness={:.2f}, effective_radius={:.1f}px".format(
            mask_info['radius_ratio'], mask_info['smoothness'], mask_info['effective_radius_pixels']))

    def apply_augmentation(self, x):
        return self.augmentor.apply_augmentation(x)

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']
        metrics_test = [[], [], [], []]

        # print("Student model: ", self.student_model)
        # print("Teacher model: ", self.teacher_model)

        for batch, data in enumerate(tqdm(self.target_test_loader, desc="Processing batches", ncols=100)):
            x, y, names = data['data'], data['mask'], data['name']
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.long)
            name_list = [str(n) for n in np.array(names).reshape(-1)]

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            self.student_model.eval()
            self.teacher_model.eval()
            self.prompt.train()

            # Train for n iterations
            for tr_iter in range(self.iters):
                # Student forward with prompt
                prompt_x, _ = self.prompt(x)
                student_logits = self.student_model(prompt_x)

                # Teacher forward with augmented input
                augmented_x = self.apply_augmentation(x)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(augmented_x)

                losses = self.criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    prompt_images=prompt_x
                )
                total_loss = losses['total']

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.optimizer.step()

                if tr_iter == 0:
                    mask_info = self.prompt.get_mask_info()
                    loss_info = f"Batch {batch}, Total Loss: {losses['total'].item():.4f}, "
                    loss_info += f"Con: {losses['con'].item():.4f}, "
                    loss_info += f"CPS: {losses['cps'].item():.4f}, "
                    if 'sar' in losses:
                        loss_info += f"SAR: {losses['sar'].item():.4f}, "
                    loss_info += f"Mask radius: {mask_info['effective_radius_pixels']:.1f}px"
                    print(loss_info)

            self.student_model.eval()
            self.prompt.eval()

            with torch.no_grad():
                prompt_x, _ = self.prompt(x)
                student_logits = self.student_model(prompt_x)

            save_images(prompt_x, name_list, self.prompt_image_dir)

            if config.model == 'segformer':
                seg_output = F.softmax(student_logits[0].permute(1, 2, 0), dim=-1).detach().cpu().numpy().argmax(axis=-1)
                save_predictions(np.expand_dims(seg_output, axis=0), name_list, self.predict_dir)
                metrics = calculate_metrics(seg_output, y.detach().cpu(), config.model)
            else:
                seg_output = torch.sigmoid(student_logits)
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu(), config.model)

            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        # Print final results
        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics: ", print_test_metric_mean)
        print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='RIM_ONE_r3',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Teacher-Student parameters
    parser.add_argument('--alpha_teacher', type=float, default=0.99, help='EMA coefficient for teacher model')
    parser.add_argument('--lambda_consistency', type=float, default=1, help='Weight for consistency loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='Weight for entropy loss')

    # Hyperparameters
    parser.add_argument('--prompt_alpha', type=float, default=0.01)

    # SAR loss
    parser.add_argument('--lambda_sar', type=float, default=0.01, help='Weight for SAR loss')
    parser.add_argument('--sar_text', type=str, default='a photo of a retinal fundus with optic disc and cup',
                        help='Text prompt for SAR alignment')

    # Augmentation
    parser.add_argument('--aug_type', type=str, default='tta',
                        help='Augmentation type: tta/spatial/fourier/style_strong/style_weak/bezier/combined')

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--dataset_root', type=str, default='../Datasets/Fundus')

    # Cuda
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='segformer')
    parser.add_argument('--epoch', type=str, default='100')



    config = parser.parse_args()

    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = DSATA(config)
    TTA.run()