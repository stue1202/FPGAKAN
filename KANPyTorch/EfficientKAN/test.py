import torch
def build_grid(grid_range, grid_size, spline_order):
    h = (grid_range[1] - grid_range[0]) / grid_size
    grid = (
        (
            torch.arange(
                -spline_order, grid_size + spline_order + 1
            )
            * h
            + grid_range[0]
        )
    )
    print("grid: ",grid)
def calculate_b_spline_bases(x):
    grid: torch.Tensor = (
        build_grid([1,-1],5,3)
    )  # (in_features, grid_size + 2 * spline_order + 1)
    #print("grid: ",grid.shape)
    #print("x: ",x.shape)
    spline_order = 3
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        print("bases: ",bases.shape)
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )
    print("bases: ",bases.shape)
    return bases
calculate_b_spline_bases(torch.tensor([1,2,3,4,5]))