# Modified for the PanPrompt3D release: packaging/import cleanup.
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib
import torch
import numpy as np
import os
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator


class Dataset_Union_ALL(Dataset):

    def __init__(
        self,
        paths,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=500,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx
        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])
        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            print("Origin diff")
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            print("Direction diff")
            sitk_image.SetDirection(sitk_label.GetDirection())
        image = sitk.GetArrayFromImage(sitk_image)
        label = sitk.GetArrayFromImage(sitk_label)
        image = image.transpose(2, 1, 0)
        label = label.transpose(2, 1, 0)
        sitk_image_arr, _ = sitk_to_nib(sitk.GetImageFromArray(image))
        sitk_label_arr, _ = sitk_to_nib(sitk.GetImageFromArray(label))
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
            label=tio.LabelMap(tensor=sitk_label_arr),
        )
        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)
        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
        if self.pcc:
            print("using pcc setting")
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                print("CropOrPad操作前 - 图像形状:", subject.image.data.shape)
                print("CropOrPad操作前 - 标签形状:", subject.label.data.shape)
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)
                print("CropOrPad操作后 - 图像形状:", subject.image.data.shape)
                print("CropOrPad操作后 - 标签形状:", subject.label.data.shape)
        import torch.nn.functional as F

        if subject.image.data.shape != subject.label.data.shape:
            image = subject.image.data
            label = subject.label.data
            min_shape = tuple(
                (
                    min(img_dim, lbl_dim)
                    for img_dim, lbl_dim in zip(image.shape, label.shape)
                )
            )
            max_shape = tuple(
                (
                    max(img_dim, lbl_dim)
                    for img_dim, lbl_dim in zip(image.shape, label.shape)
                )
            )
            cropped_image = image[
                : min_shape[0], : min_shape[1], : min_shape[2], : min_shape[3]
            ]
            cropped_label = label[
                : min_shape[0], : min_shape[1], : min_shape[2], : min_shape[3]
            ]
            padded_image = F.pad(
                cropped_image,
                (
                    0,
                    max_shape[3] - min_shape[3],
                    0,
                    max_shape[2] - min_shape[2],
                    0,
                    max_shape[1] - min_shape[1],
                    0,
                    max_shape[0] - min_shape[0],
                ),
                mode="constant",
                value=0,
            )
            padded_label = F.pad(
                cropped_label,
                (
                    0,
                    max_shape[3] - min_shape[3],
                    0,
                    max_shape[2] - min_shape[2],
                    0,
                    max_shape[1] - min_shape[1],
                    0,
                    max_shape[0] - min_shape[0],
                ),
                mode="constant",
                value=0,
            )
            subject.image.set_data(padded_image)
            subject.label.set_data(padded_label)
            return self.__getitem__(np.random.randint(self.__len__()))
        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.mode == "train" and self.data_type == "Tr":
            return (
                subject.image.data.clone().detach().float(),
                subject.label.data.clone().detach().long(),
            )
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "origin": sitk_label.GetOrigin(),
                "direction": sitk_label.GetDirection(),
                "spacing": sitk_label.GetSpacing(),
            }
            return (
                subject.image.data.clone().detach().float(),
                subject.label.data.clone().detach().long(),
                meta_info,
            )
        else:
            return (
                subject.image.data.clone().detach().float(),
                subject.label.data.clone().detach().long(),
                self.image_paths[index],
            )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []
        for path in paths:
            d = os.path.join(path, f"labels{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    label_path = os.path.join(
                        path, f"labels{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(
                        os.path.join(path, f"images{self.data_type}", f"{base}.nii.gz")
                    )
                    self.label_paths.append(label_path)


class Union_Dataloader(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
