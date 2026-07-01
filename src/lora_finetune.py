"""
Compare LoRA fine-tuning to full fine-tuning on the Oxford-IIIT Pet dataset.

PART 1 — LoRA module + model builder
    - LoRAConv2d: a frozen base Conv2d plus two trainable low-rank convs
      A (in_c -> r, same kernel/stride/padding as base) and B (r -> out_c, 1x1).
      Forward: y = base(x) + (alpha / r) * B(A(dropout(x))).
      A is Kaiming-initialised, B is zero — so the wrapped layer starts
      identical to the pre-trained one and the adapter only "learns" away
      from zero during training.
    - inject_lora_conv: walks the model and swaps targeted Conv2ds in place.
    - build_lora_resnet34: pre-trained ResNet34 with backbone frozen, new
      37-way fc head, and LoRA adapters on (conv1, conv2) by default.

"""

import math
import torch
import torch.nn as nn
from torchvision import models


# -----------------------
# LoRA layer
# -----------------------
class LoRAConv2d(nn.Module):
    """
    Wraps a Conv2d with a low-rank additive update.

      base:    pre-trained convolution, frozen.
      lora_A:  Conv2d(in_c -> r) with the same kernel/stride/padding/dilation
               as `base`. Maps the input into a low-rank channel space.
      lora_B:  Conv2d(r -> out_c, kernel=1) that expands back to `out_c`.

    The two together form a rank-r update to `base.weight`. Output is the
    sum of the frozen base output and the (alpha / r)-scaled adapter output.
    """

    def __init__(self, base_conv: nn.Conv2d, r: int = 8,
                 alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        in_c = base_conv.in_channels
        out_c = base_conv.out_channels

        self.lora_A = nn.Conv2d(
            in_channels=in_c,
            out_channels=r,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


# -----------------------
# Helpers
# -----------------------
def _get_parent_and_attr(root: nn.Module, dotted_name: str):
    """Resolve a dotted module path to (parent_module, attribute_name)."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_conv(model: nn.Module,
                     target_names=("conv1", "conv2", "conv3"),
                     r: int = 8,
                     alpha: int = 16,
                     dropout: float = 0.0) -> int:
    """
    Replace every Conv2d in `model` whose attribute name is in `target_names`
    with a LoRAConv2d wrapping the original conv. Returns the count replaced.
    """
    # Collect first so we don't mutate while iterating named_modules().
    to_replace = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            short_name = full_name.split(".")[-1]
            if short_name in target_names:
                to_replace.append(full_name)

    for full_name in to_replace:
        parent, attr = _get_parent_and_attr(model, full_name)
        original = getattr(parent, attr)
        setattr(parent, attr, LoRAConv2d(original, r=r, alpha=alpha, dropout=dropout))

    return len(to_replace)


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------
# Model builder
# -----------------------
def build_lora_resnet50(num_classes: int = 37,
                        r: int = 8,
                        alpha: int = 16,
                        dropout: float = 0.0,
                        target_names=("conv1", "conv2", "conv3")) -> nn.Module:
    """
    Pre-trained ResNet50 with a fresh `fc` head and LoRA adapters injected
    at the named Conv2ds. Order matters: freeze first, then swap in the new
    head (its params come out as requires_grad=True by default), then inject
    LoRA (the wrapped base stays frozen, A and B start trainable).
    """
    model = models.resnet50(weights="IMAGENET1K_V1")

    freeze_all(model)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    n_replaced = inject_lora_conv(
        model,
        target_names=target_names,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    print(f"Injected LoRA into {n_replaced} Conv2d layers "
          f"(targets={list(target_names)}, r={r}, alpha={alpha})")

    return model

def build_linear_probe(num_classes: int = 37) -> nn.Module:
    """
    Pre-trained ResNet50 with a fresh `fc` head, but no LoRA adapters.
    This is a "linear probe" baseline where only the final layer is trained.
    """
    model = models.resnet50(weights="IMAGENET1K_V1")
    freeze_all(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def build_full_finetune(num_classes: int = 37) -> nn.Module:
    """
    Pre-trained ResNet50 with a fresh `fc` head, and all backbone layers
    unfrozen. This is the standard full fine-tuning baseline.
    """
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
# -----------------------
# Self-test (Part 1 sanity check)
# -----------------------
if __name__ == "__main__":
    model = build_lora_resnet50(num_classes=37, r=8, alpha=16)

    trainable = count_trainable(model)
    total = count_total(model)
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable fraction: {trainable / total:.4%}")

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}  (expected (2, 37))")
    assert y.shape == (2, 37)

    # Confirm the backbone Conv2d weights are frozen and only LoRA A/B + fc
    # are trainable.
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameter groups (first 10):")
    for n in trainable_names[:10]:
        print(f"  {n}")
    print(f"  ... ({len(trainable_names)} trainable tensors total)")
