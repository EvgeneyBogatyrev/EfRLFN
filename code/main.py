import torch
from code.model import EfRLFN



if __name__ == "__main__":
    model = EfRLFN(upscale=4)
    
    random_input = torch.rand(1, 3, 320, 480)

    upscaled_result = model(random_input)
    print("The size of am output is", upscaled_result.size())

