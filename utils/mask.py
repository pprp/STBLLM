import torch

"""
Generate the structural mask on the basis of the split border
"""

# 3mask
# def generate_structural_mask(origin_matrix, mask3, braq1_border):
#     mask1_2 = ~mask3

#     binary_group = torch.abs(origin_matrix * mask1_2)

#     mask2 = binary_group >= braq1_border
#     mask1 = binary_group < braq1_border

#     mask1 = mask1 * mask1_2
#     mask2 = mask2 * mask1_2

#     return mask1, mask2

# 4mask
def generate_structural_mask(origin_matrix, mask4, braq1_border, braq2_border):
    mask1_2_3 = ~mask4

    binary_group = torch.abs(origin_matrix * mask1_2_3)

    mask3 = binary_group >= braq2_border
    mask2 = (binary_group >= braq1_border) & (binary_group < braq2_border)
    mask1 = binary_group < braq1_border

    mask1 = mask1 * mask1_2_3
    mask2 = mask2 * mask1_2_3
    mask3 = mask3 * mask1_2_3

    return mask1, mask2, mask3


def generate_mask(origin_matrix, braq2_border, braq1_border):
    mask3 = torch.abs(origin_matrix) >= braq2_border
    mask1 = torch.abs(origin_matrix) <= braq1_border
    mask2 = (torch.abs(origin_matrix) > braq1_border) & (
        torch.abs(origin_matrix) < braq2_border
    )
    return mask1, mask2, mask3
