import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats


class CustomWanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "mask_image": ("IMAGE", ),
                             "first_frame": ("IMAGE", ),
                             "last_frame": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/custom_video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, mask_image=None, first_frame=None, last_frame=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            
            # If first_frame is provided, replace the first frame of start_image
            if first_frame is not None:
                first_frame = comfy.utils.common_upscale(first_frame.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                if start_image.shape[0] > 0:
                    start_image[0] = first_frame[0]
                    # start_image[1:] = 0.5
            
            # If last_frame is provided, replace the last four frames of start_image
            if last_frame is not None and start_image.shape[0] >= 4:
                last_frame = comfy.utils.common_upscale(last_frame.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                for i in range(4):
                    if start_image.shape[0] - i - 1 >= 0:
                        start_image[start_image.shape[0] - i - 1] = last_frame[0]
            
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            # Process mask_image input
            if mask_image is not None:
                # Process and scale mask_image
                mask_image = comfy.utils.common_upscale(mask_image[:length].movedim(-1, 1), width // 8, height // 8, "bilinear", "center").movedim(1, -1)
                
                # Convert mask_image to grayscale and binarize
                if mask_image.shape[-1] > 1:  # If it is a color image
                    mask_gray = 0.299 * mask_image[..., 0] + 0.587 * mask_image[..., 1] + 0.114 * mask_image[..., 2]
                else:
                    mask_gray = mask_image[..., 0]
                
                # Normalize to 0-1 range
                if mask_gray.dtype == torch.uint8:
                    mask_gray = mask_gray.float() / 255.0
                
                # Binarize
                mask_binary = (mask_gray > 0.5).float()
                
                # Create mask tensor
                mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
                
                # Correctly map mask_binary to latent frames
                # If latent has N frames, then mask_binary has 4N+1 frames
                # The first frame of mask_binary corresponds to the first frame of latent, then every 4 frames of mask_binary correspond to one frame of latent
                mask[0, 0, 0] = mask_binary[0]  # The first frame corresponds directly
                
                num_latent_frames = latent.shape[2]
                for latent_idx in range(1, num_latent_frames):
                    # For each latent frame, find the corresponding mask_binary frame index
                    # From the second frame, each latent frame corresponds to 4 frames in mask
                    mask_start_idx = 1 + (latent_idx - 1) * 4
                    
                    # Ensure not to exceed the range of mask_binary
                    if mask_start_idx < mask_binary.shape[0]:
                        # Use the mask frame at this position
                        mask[0, 0, latent_idx] = mask_binary[mask_start_idx]
                
                # If first_frame is provided, set the mask of the first frame to 0
                if first_frame is not None:
                    mask[0, 0, 0] = 0.0
                
                # If last_frame is provided, set the mask of the last frame to 0
                if last_frame is not None and num_latent_frames > 0:
                    mask[0, 0, -1] = 0.0
                
                # mask[:, :, ((start_image.shape[0] - 1) // 4) + 1:] = 1.0
                # mask[:] = 1.0
                # latent[:] = 0.0
            else:
                # Default mask behavior
                mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
                mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0
                
                # If first_frame is provided, set the mask of the first frame to 0
                if first_frame is not None:
                    mask[0, 0, 0] = 0.0
                
                # If last_frame is provided, set the mask of the last frame to 0
                if last_frame is not None and latent.shape[2] > 0:
                    mask[0, 0, -1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "CustomWanImageToVideo": CustomWanImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomWanImageToVideo": "Custom Image to Video Node",
} 