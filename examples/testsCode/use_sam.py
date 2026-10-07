from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](
    checkpoint="sam_vit_b_01ec64.pth"
)

sam.eval()

# 冻结 SAM
for param in sam.image_encoder.parameters():
    param.requires_grad = False