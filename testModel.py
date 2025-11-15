from core.igev_lite import IGEVStereoLite
import torch


if __name__ == '__main__':
    model = IGEVStereoLite(max_disp=192)
    model.eval()
    input_data = torch.randn(1, 3, 256, 512)
    output = model(input_data)

    if isinstance(output, tuple):
        init_disp, disp_preds = output
        print(f"Initial disparity shape: {init_disp.shape}")
        print(f"Number of iterative predictions: {len(disp_preds)}")
        for i, pred in enumerate(disp_preds):
            print(f"Iteration {i + 1} prediction shape: {pred.shape}")
    else:
        print(f"Final disparity prediction shape: {output.shape}")