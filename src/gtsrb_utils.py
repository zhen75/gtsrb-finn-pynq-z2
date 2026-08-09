import torch
import random
import numpy as np

from pathlib import Path
from quant_vgg import QuantVgg

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from torchvision import transforms
from torchvision.datasets import GTSRB


def get_gtsrb_loaders(
    data_root="./data", batch_size=32, num_workers=4, val_ratio=0.1, seed=42
):
    mean = [0.3399, 0.3121, 0.3214]

    std = [0.2760, 0.2625, 0.2690]

    train_transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.2,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    base_dataset = GTSRB(root=data_root, split="train", download=True)

    total_size = len(base_dataset)

    val_size = int(total_size * val_ratio)

    train_size = total_size - val_size

    # ==============================
    # 修改开始：按 (class_id, sequence_id) 划分 train/val
    # ==============================

    import random
    from collections import defaultdict

    # (类别, 序列) -> 图片索引
    group_to_indices = defaultdict(list)

    for idx, (path, class_id) in enumerate(base_dataset._samples):

        filename = path.split("/")[-1]

        # 例如：
        # 00000_00001.ppm
        # sequence_id = 00000
        sequence_id = filename.split("_")[0]

        # 关键修改：
        # 分组键加入类别
        group_key = (class_id, sequence_id)

        group_to_indices[group_key].append(idx)

    # 按类别保存所有sequence
    class_to_groups = defaultdict(list)

    for group_key in group_to_indices.keys():

        class_id, sequence_id = group_key

        class_to_groups[class_id].append(group_key)

    rng = random.Random(seed)

    train_indices = []
    val_indices = []

    # 每个类别独立抽10%的sequence
    for class_id, groups in class_to_groups.items():

        rng.shuffle(groups)

        # 当前类别验证sequence数量
        num_val_groups = max(1, int(len(groups) * val_ratio))

        val_groups = set(groups[:num_val_groups])

        for group in groups:

            if group in val_groups:

                val_indices.extend(group_to_indices[group])

            else:

                train_indices.extend(group_to_indices[group])

        # ==============================
        # 修改结束：按 (class_id, sequence_id) 划分 train/val
        # ==============================

    train_dataset_full = GTSRB(
        root=data_root, split="train", download=False, transform=train_transform
    )

    train_dataset = Subset(train_dataset_full, train_indices)

    target_multipliers = {21: 2.0, 30: 1.5, 41: 2.0}
    sample_weights = [
        target_multipliers.get(base_dataset._samples[idx][1], 1.0)
        for idx in train_indices
    ]
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    val_dataset_full = GTSRB(
        root=data_root, split="train", download=False, transform=test_transform
    )

    val_dataset = Subset(val_dataset_full, val_indices)

    test_dataset = GTSRB(
        root=data_root, split="test", download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ==============================
    # 修改开始：计算 class weight
    # ==============================

    import numpy as np

    # 只统计训练集类别
    train_targets = [base_dataset._samples[i][1] for i in train_indices]

    class_counts = np.bincount(train_targets, minlength=43)

    # 方法1：反频率
    # class_weights = 1.0 / class_counts

    # 方法2：更温和的反频率权重
    class_weights = class_counts**-0.25

    # 归一化，让平均权重=1
    class_weights = class_weights / class_weights.mean()

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("class counts:")
    print(class_counts)

    print("class weights:")
    print(class_weights)

    # ==============================
    # 修改结束：计算 class weight
    # ==============================

    return (train_loader, val_loader, test_loader, class_weights)


def get_npy(model, dataloader, target):
    images_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            quantized = model.quant_input(images)
            integer_codes = quantized.int(float_datatype=True)

            integer_codes = integer_codes.permute(0, 2, 3, 1)

            images_list.append(integer_codes.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    inputs_path = target / "test_inputs_int8_w4a4.npy"
    labels_path = target / "test_labels.npy"

    np.save(inputs_path, images)
    np.save(labels_path, labels)


def get_expected_outputs(model, dataloader, target):
    outputs_list = []
    predictions_list = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            outputs_list.append(outputs.cpu().numpy())
            predictions_list.append(predictions.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

        expected_outputs = np.concatenate(outputs_list, axis=0).astype(np.float32)
        predicted_labels = np.concatenate(predictions_list, axis=0).astype(np.int64)
        true_labels = np.concatenate(labels_list, axis=0).astype(np.int64)

        np.save(
            target / "test_expected_w4a4.npy",
            expected_outputs,
        )
        np.save(
            target / "test_pred_labels_w4a4.npy",
            predicted_labels,
        )

        saved_labels = np.load(target / "test_labels.npy")
        assert np.array_equal(true_labels, saved_labels)

        accuracy = np.mean(predicted_labels == true_labels)

        print("Expected output shape:", expected_outputs.shape)
        print("Prediction shape:", predicted_labels.shape)
        print("Test accuracy:", accuracy)


if __name__ == "__main__":
    target = Path("validation_npy")
    target.mkdir(parents=True, exist_ok=True)

    weight_path = "qat_experiments/experiment_5/w4a4_qat.pth"

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
    )

    model = QuantVgg(
        weight_bit=4,
        activate_bit=4,
        dropout=0.2,
        class_weights=torch.ones(43),
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _, _, test_loader, _ = get_gtsrb_loaders()

    get_npy(
        model=model,
        dataloader=test_loader,
        target=target,
    )

    get_expected_outputs(
        model=model,
        dataloader=test_loader,
        target=target,
    )
