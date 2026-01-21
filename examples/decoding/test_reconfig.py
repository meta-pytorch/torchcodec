from torchcodec.decoders import set_cuda_backend, VideoDecoder


import torch
from PIL import Image
print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.cuda.get_device_properties(0)=}")


def plot_cpu_and_cuda_frames(cpu_frames: torch.Tensor, cuda_frames: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("Cannot plot, please run `pip install matplotlib pillow`")
        return

    # 假设 frames 是 [N, C, H, W]，值范围 [0, 255] 或 [0.0, 1.0]
    def tensor_to_pil(tensor):
        # 确保在 CPU 上
        img = tensor.cpu()
        # 如果是 float 类型且最大值 <= 1.0，乘以 255
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            if img.max() <= 1.0:
                img = (img * 255).clamp(0, 255).to(torch.uint8)
            else:
                img = img.clamp(0, 255).to(torch.uint8)
        else:
            img = img.to(torch.uint8)
        # 转为 [H, W, C]
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        elif img.ndim == 4:
            raise ValueError("Expected single image, got batch")
        return Image.fromarray(img.numpy())

    try:
        n_rows = cpu_frames.shape[0]  # 替代未定义的 `indices`
        fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
        if n_rows == 1:
            axes = [axes]  # 统一为 list 避免索引错误

        for i in range(n_rows):
            axes[i][0].imshow(tensor_to_pil(cpu_frames[i]))
            axes[i][1].imshow(tensor_to_pil(cuda_frames[i]))

        axes[0][0].set_title("CPU decoder", fontsize=24)
        axes[0][1].set_title("CUDA decoder", fontsize=24)
        plt.setp(axes, xticks=[], yticks=[])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed (skipping): {e}")





def convert_tensor2image(tensor: torch.Tensor, prefix: str = "right_images"):
    for i in range(tensor.size(0)):
        # Take first frame
        frame = tensor[i]

        # Normalize layout to HWC
        if frame.ndim != 3:
            raise RuntimeError(f"Unexpected frame ndim={frame.ndim}, shape={tuple(frame.shape)}")

        # If CHW -> HWC
        if frame.shape[0] in (1, 3, 4) and frame.shape[0] < frame.shape[-1]:
            frame = frame.permute(1, 2, 0)

        # Move to CPU
        frame = frame.detach().to("cpu")

        # Convert dtype to uint8 for image saving
        if frame.dtype != torch.uint8:
            # common cases: float in [0,1] or [0,255]
            fmin = float(frame.min())
            fmax = float(frame.max())
            print(f"  value range before uint8: min={fmin:.6f}, max={fmax:.6f}")

            if fmax <= 1.0 + 1e-6:
                frame = (frame * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                frame = frame.clamp(0, 255).to(torch.uint8)

        np_img = frame.numpy()
        print("  numpy shape:", np_img.shape)
        print("  numpy dtype:", np_img.dtype)
        print("  numpy value range: min={}, max={}".format(np_img.min(), np_img.max()))

        # If RGBA -> RGB (optional)
        if np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]

        out_path = f"./{prefix}/decoded_check_{i}.png"
        Image.fromarray(np_img).save(out_path)
        print(f"Saved decoded image to: {out_path}")

video_file = "video.mp4"
video_file2 = "video2.mp4"



with set_cuda_backend("beta"):  # Use the BETA backend, it's faster!
    decoder = VideoDecoder(video_file2, device="cuda", seek_mode="exact")
indices = list(range(0, decoder.metadata.num_frames, int(decoder.metadata.average_fps)))
print(indices)
cuda_frames_ref = decoder.get_frames_at(indices).data
# convert_tensor2image(cuda_frames_ref, "right_images")
# print(decoder.metadata)

with set_cuda_backend("beta"):  # Use the BETA backend, it's faster!
    decoder = VideoDecoder(video_file2, device="cuda", seek_mode="exact")
cuda_frames_exp = decoder.get_frames_at(indices).data
# convert_tensor2image(cuda_frames_exp, "test_images")
# print(decoder.metadata)


plot_cpu_and_cuda_frames(cuda_frames_ref, cuda_frames_exp)


frames_equal = torch.equal(cuda_frames_ref.to("cuda"), cuda_frames_exp)
mean_abs_diff = torch.mean(
    torch.abs(cuda_frames_ref.float().to("cuda") - cuda_frames_exp.float())
)
max_abs_diff = torch.max(torch.abs(cuda_frames_ref.to("cuda").float() - cuda_frames_exp.float()))
print(f"{frames_equal=}")
print(f"{mean_abs_diff=}")
print(f"{max_abs_diff=}")

