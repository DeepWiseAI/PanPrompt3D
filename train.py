# Full PanPrompt3D training, derived from train_all_datasets_respacing15.py.
# Modified: portable paths, full-model-only imports, tensor-only initialization,
# CPU-only loader support and argument parsing. Historical numerical operations
# (11 interactions, seed behavior, accumulation and CF loss) are retained.
import numpy as np
import random
import datetime
import logging
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from panprompt3d.build_model import sam_model_registry3D
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from panprompt3d.clicks import get_next_click3D_torch_2
from panprompt3d.data import Dataset_Union_ALL, Union_Dataloader
from panprompt3d.losses import CF_Loss_3D
import gc

all_train_datasets = []
parser = argparse.ArgumentParser(description="Train the full PanPrompt3D model")
parser.add_argument(
    "--train_roots",
    nargs="+",
    required=True,
    help="One or more roots containing imagesTr/ and labelsTr/",
)
parser.add_argument("--task_name", type=str, default="union_train")
parser.add_argument("--click_type", choices=["random"], default="random")
parser.add_argument("--multi_click", action="store_true", default=False)
parser.add_argument("--num_clicks_max", type=int, default=30)
parser.add_argument("--model_type", choices=["vit_b_ori"], default="vit_b_ori")
parser.add_argument(
    "--checkpoint", required=True, help="Tensor-only initialization checkpoint"
)
parser.add_argument("--device", choices=["cuda"], default="cuda")
parser.add_argument("--work_dir", type=str, default="work_dir")
parser.add_argument("--which_data_aug", choices=["v1", "v2", "v3"], default="v1")
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
parser.add_argument("--multi_gpu", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--lr_scheduler", choices=["multisteplr"], default="multisteplr")
parser.add_argument("--step_size", type=int, nargs="+", default=[120, 180])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--img_size", type=int, choices=[128], default=128)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--accumulation_steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.0008)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--port", type=int, default=12318)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--use_cf_loss", action="store_true", default=True)
parser.add_argument("--alpha_cf_loss", type=float, default=0.5)
parser.add_argument("--gamma_cf_loss", type=float, default=0.5)
args = parser.parse_args()
all_train_datasets = args.train_roots
if not args.resume and not os.path.isfile(args.checkpoint):
    parser.error("Initialization checkpoint not found")
for root in all_train_datasets:
    if not os.path.isdir(os.path.join(root, "imagesTr")) or not os.path.isdir(
        os.path.join(root, "labelsTr")
    ):
        parser.error("Each training root must contain imagesTr and labelsTr")
device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.gpu_ids])

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {"random": get_next_click3D_torch_2}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    if args.which_data_aug == "v1":
        train_dataset = Dataset_Union_ALL(
            paths=all_train_datasets,
            transform=tio.Compose(
                [
                    tio.ToCanonical(),
                    tio.CropOrPad(
                        mask_name="label",
                        target_shape=(args.img_size, args.img_size, args.img_size),
                    ),
                    tio.RandomFlip(axes=(0, 1, 2)),
                ]
            ),
            threshold=1000,
        )
    elif args.which_data_aug == "v2":
        train_dataset = Dataset_Union_ALL(
            paths=all_train_datasets,
            transform=tio.Compose(
                [
                    tio.ToCanonical(),
                    tio.CropOrPad(
                        mask_name="label",
                        target_shape=(args.img_size, args.img_size, args.img_size),
                    ),
                    tio.RandomFlip(axes=(0, 1, 2)),
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True),
                    tio.RandomNoise(mean=0, std=(0.1, 0.5)),
                    tio.RandomBlur(std=(0.5, 1.5)),
                ]
            ),
            threshold=1000,
        )
    elif args.which_data_aug == "v3":
        train_dataset = Dataset_Union_ALL(
            paths=all_train_datasets,
            transform=tio.Compose(
                [
                    tio.ToCanonical(),
                    tio.CropOrPad(
                        mask_name="label",
                        target_shape=(args.img_size, args.img_size, args.img_size),
                    ),
                    tio.RandomFlip(axes=(0, 1, 2)),
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True),
                    tio.RandomGhosting(intensity=(0.5, 1.0), p=0.5),
                ]
            ),
            threshold=1000,
        )
    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"get_dataloaders: {len(train_dataloader)}")
    return train_dataloader


class BaseTrainer:

    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.resume:
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, "sam_model_latest.pth")
            )
        else:
            self.init_checkpoint(self.args.checkpoint)
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
        self.optimizer = torch.optim.AdamW(
            [
                {"params": sam_model.image_encoder.parameters()},
                {
                    "params": sam_model.prompt_encoder.parameters(),
                    "lr": self.args.lr * 0.1,
                },
                {
                    "params": sam_model.mask_decoder.parameters(),
                    "lr": self.args.lr * 0.1,
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.args.step_size, self.args.gamma
            )
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.args.step_size[0], self.args.gamma
            )
        elif self.args.lr_scheduler == "coswarm":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer
            )
        elif self.args.lr_scheduler == "cosineannealinglr":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(
                    ckp_path,
                    map_location=self.args.device,
                    weights_only=not self.args.resume,
                )
            else:
                last_ckpt = torch.load(
                    ckp_path,
                    map_location=self.args.device,
                    weights_only=not self.args.resume,
                )
        if last_ckpt:
            model = self.model.module if self.args.multi_gpu else self.model
            match = model.load_state_dict(
                last_ckpt.get("model_state_dict", last_ckpt), strict=False
            )
            print(
                f"Initialization compatibility: {len(match.missing_keys)} missing, {len(match.unexpected_keys)} extra model keys"
            )
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt["epoch"]
                self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(last_ckpt["lr_scheduler_state_dict"])
                self.losses = last_ckpt["losses"]
                self.dices = last_ckpt["dices"]
                self.best_loss = last_ckpt["best_loss"]
                self.best_dice = last_ckpt["best_dice"]
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": all_train_datasets,
            },
            join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"),
        )

    def batch_forward(
        self, sam_model, image_embedding, gt3D, low_res_masks, points=None
    ):
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points, boxes=None, masks=low_res_masks
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        prev_masks = F.interpolate(
            low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False
        )
        return (low_res_masks, prev_masks)

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](
            prev_masks, gt3D
        )
        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)
        self.click_points.append(points_co)
        self.click_labels.append(points_la)
        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)
        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return (points_input, labels_input)

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(args.img_size // 4, args.img_size // 4, args.img_size // 4),
        )
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)
            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None
                )
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D,
                    low_res_masks,
                    points=[points_input, labels_input],
                )
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return (prev_masks, return_loss)

    def get_dice_score(self, prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5
            mask_pred = mask_pred > mask_threshold
            mask_gt = mask_gt > 0
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.nan
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = prev_masks > 0.5
        true_masks = gt3D > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders
        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        step = 0
        for step, (image3D, gt3D) in enumerate(tbar):
            my_context = (
                self.model.no_sync
                if self.args.rank != -1 and step % self.args.accumulation_steps != 0
                else nullcontext
            )
            with my_context():
                image3D = self.norm_transform(image3D.squeeze(dim=1))
                image3D = image3D.unsqueeze(dim=1)
                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with torch.cuda.amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)
                    self.click_points = []
                    self.click_labels = []
                    pred_list = []
                    prev_masks, loss = self.interaction(
                        sam_model, image_embedding, gt3D, num_clicks=11
                    )
                    if self.args.use_cf_loss:
                        cf_loss_ = CF_Loss_3D(
                            img_depth=image3D.shape[-1],
                            beta=0.0,
                            gamma=self.args.gamma_cf_loss,
                        )
                        cf_loss_value = cf_loss_(prev_masks, gt3D)
                        seg_loss = loss
                        loss = seg_loss + self.args.alpha_cf_loss * cf_loss_value
                if loss.dim() > 0:
                    loss = loss.mean()
                loss /= self.args.accumulation_steps
                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()
                self.scaler.scale(loss).backward()
            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}"
                    )
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(
                                epoch,
                                sam_model.state_dict(),
                                describe=f"{epoch}_step_dice:{print_dice}_best",
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
        epoch_loss /= step + 1
        epoch_dice /= step + 1
        return (epoch_loss, epoch_iou, epoch_dice, pred_list)

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel("Epoch")
        plt.ylabel(f"{save_name}")
        plt.savefig(join(MODEL_SAVE_PATH, f"{save_name}.png"))
        plt.close()

    def train(self):
        self.scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")
            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, self.args.num_clicks_max + 1)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(
                epoch, num_clicks
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f"EPOCH: {epoch}, Loss: {epoch_loss} \n")
                print(f"EPOCH: {epoch}, Dice: {epoch_dice} \n")
                logger.info(
                    f"Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}\n"
                )
                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                self.save_checkpoint(epoch, state_dict, describe="latest")
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe="loss_best")
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe="dice_best")
                self.plot_result(self.losses, "Dice + Cross Entropy + 0.5 VD ", "Loss")
                self.plot_result(self.dices, "Dice", "Dice")
        logger.info(
            "====================================================================="
        )
        logger.info(f"ahpha cf loss: {self.args.alpha_cf_loss}")
        logger.info(f"Best loss: {self.best_loss}")
        logger.info(f"Best dice: {self.best_dice}")
        logger.info(f"Total loss: {self.losses}")
        logger.info(f"Total dice: {self.dices}")
        logger.info(
            "====================================================================="
        )
        logger.info(f"args : {self.args}")
        logger.info(f"Used datasets : {all_train_datasets}")
        logger.info(
            "====================================================================="
        )


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            if args.device == "mps":
                args.device = torch.device("mps")
            else:
                args.device = torch.device("cuda:0")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy("file_system")
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        dataloaders = get_dataloaders(args)
        model = build_model(args)
        trainer = BaseTrainer(model, dataloaders, args)
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    init_seeds(args.seed + rank)
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode="w",
        filename=os.path.join(LOG_OUT_DIR, f"output_{cur_time}.log"),
    )
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{args.port}",
        world_size=world_size,
        rank=rank,
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
