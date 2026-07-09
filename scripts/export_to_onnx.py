import argparse
import logging
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
from CorridorKeyModule.inference_engine import CorridorKeyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_onnx(checkpoint_path: str, output_path: str, img_size: int = 2048):
    logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    
    # We set CORRIDORKEY_SKIP_COMPILE=1 to avoid compiling the model for export
    import os
    os.environ["CORRIDORKEY_SKIP_COMPILE"] = "1"
    
    engine = CorridorKeyEngine(
        checkpoint_path=checkpoint_path,
        device="cpu",
        img_size=img_size,
        mixed_precision=False,
    )
    
    model = engine.model
    model.eval()
    
    # Create dummy input: 4 channels (3 for RGB image, 1 for linear mask)
    dummy_input = torch.randn(1, 4, img_size, img_size, dtype=torch.float32)
    
    logger.info(f"Exporting model to {output_path}...")
    
    class Wrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
            
        def forward(self, x):
            res = self.core(x)
            return res["alpha"], res["fg"]
            
    wrapped_model = Wrapper(model)
    wrapped_model.eval()
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['alpha', 'fg'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'alpha': {0: 'batch_size'},
            'fg': {0: 'batch_size'}
        }
    )
    logger.info(f"Export completed successfully. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CorridorKey model to ONNX format")
    parser.add_argument("checkpoint", type=str, help="Path to input .safetensors or .pth file")
    parser.add_argument("output", type=str, help="Path to output .onnx file")
    parser.add_argument("--img-size", type=int, default=2048, help="Image size (default: 2048)")
    
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output, args.img_size)
