import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
import einops
import math
from ..helpers import render_junctions
class JunctionFunctions(render_junctions.JunctionRenderer):
    """Maps Junctions To Images."""

    def __init__(self,
                 opts: ConfigDict,
                 input_opts: ConfigDict):

        # First, define parameters that depend on opts
        self.patchmin = opts.patchmin
        self.patchmax = opts.patchmax
        self.patchsize = opts.patchsize
        self.stride = opts.stride
        self.num_wedges = opts.num_wedges
        self.delta = opts.delta
        self.eta = opts.eta

        self.mask_shape = opts.mask_shape
        self.jparameterization = opts.jparameterization
        self.bparameterization = opts.bparameterization
        self.patch_scales = opts.patch_scales

        # Next, define parameters that depend on input_opts
        self.height = input_opts.height
        self.width = input_opts.width
        self.channels = input_opts.channels
        self.hpatches = input_opts.hpatches
        self.wpatches = input_opts.wpatches

        self.val_features = [1, self.channels, self.height, self.width]
        self.val_boundaries = [1, 1, self.height, self.width]
        self.patch_density = self.fold(torch.ones([1, 1,
                                                   self.patchsize, self.patchsize,
                                                   self.hpatches, self.wpatches]),
                                       self.val_boundaries)

    def unfold(self,
               im: torch.Tensor,
               patchsize: int = 17,
               stride: int = 1) -> torch.Tensor:
        """Extract patches from an image.

        Args:
          im: Tensor of shape [N, C, H, W] to be unfolded into patches
          patchsize: Size of extracted patches
          stride: Stride of extracted patches

        Returns:
          Tensor of shape [N, C, R, R, H', W'] containing all extracted patches.
        """
        if im.shape[1] not in {1, 3}:
            im = im.permute(0, 3, 1, 2)  # (n, h, w, c) -> (n, c, h, w)

        # Ensure the image shape is (n, c, h, w)
        assert im.shape[1] in {1, 3}, f"Unexpected shape after transpose: {im.shape}"
        _, channels, height, width = im.shape

        # Define patchsize and stride and use to calculate the number of patches for each axis
        patchsize = self.patchsize if patchsize is None else patchsize
        stride = self.stride if stride is None else stride

        hpatches = self.hpatches if (patchsize is None) and (stride is None) else (
            (height - patchsize) // stride + 1)
        wpatches = self.wpatches if (patchsize is None) and (stride is None) else (
            (width - patchsize) // stride + 1)
        # print('edffggggg')
        # print(hpatches)
        # print(patchsize)
        patches = im.unfold(2, patchsize, stride).unfold(3, patchsize, stride)
        patches = patches.permute(0, 1, 4, 5, 2, 3).contiguous()
        patches = patches.view(-1, channels, patchsize, patchsize, hpatches, wpatches)

        return patches

    def fold(self,
             unfolded,
             output_shape,
             fn='sum',
             stride=1):
        """Fold patches of an image using function fn.

        Args:
          unfolded: Tensor of shape [N, C, R, R, H', W'] containing all unfolded patches.
          output_shape: Shape of folded array.
          fn: Function to fold with (example: sum, mean, etc.)
          stride: Stride of unfolded patches.

        Returns:
          Tensor of shape [N, C, H, W]
        """
        batch_size, channels, patchsize, _, hpatches, wpatches = unfolded.shape
        stride = self.stride if stride is None else stride
        height, width = output_shape[2], output_shape[3]

        output = torch.zeros(output_shape).to(unfolded.device)
        count = torch.zeros(output_shape).to(unfolded.device)

        for i in range(0, hpatches):
            for j in range(0, wpatches):
                output[:, :, i * stride:i * stride + patchsize, j * stride:j * stride + patchsize] += \
                    unfolded[:, :, :, :, i, j]
                count[:, :, i * stride:i * stride + patchsize, j * stride:j * stride + patchsize] += 1

        if fn == 'mean':
            output = output / count

        return output

    def local2global(self,
                     local_features: torch.Tensor,
                     patch_density: torch.Tensor,
                     stride: int = 1) -> torch.Tensor:
        """Takes feature patches and folds and normalizes to form global features.

        Args:
          local_features: Tensor of shape [N, C, R, R, H', W'].
          patch_density: Number of patches that overlap each pixel. Used for normalization.
          stride: Stride of the patches.

        Returns:
          Tensor containing folded global features
        """
        print("local_features device:", local_features.device)
        print("patch_density device:", patch_density.device)
        batch, channels, patchsize, _, hpatches, wpatches = local_features.shape

        height = hpatches * stride + patchsize - 1
        width = wpatches * stride + patchsize - 1

        val_outputs = [batch, channels, height, width]
        # val_outputs.to('GPU')
        # patch_density.to('GPU')
        # val_outputs = torch.tensor(val_outputs, device=local_features.device)

        # Ensure patch_density is on the same device
        patch_density = patch_density.to(local_features.device)
        print("val_outputs device:", torch.tensor(val_outputs).device)
        global_outputs = self.fold(local_features, val_outputs, stride=stride) / (patch_density + 1e-5)

        return global_outputs

    def get_avg_wedge_feature(self,
                              input_features: torch.Tensor,
                              global_features: torch.Tensor,
                              wedges: torch.Tensor,
                              patchsize: int = None,
                              stride: int = None,
                              lmbda_wedge_mixing: float = 0.0):
        """Find smoothed patches of the image along with wedge colors.

        Args:
          input_features: Input features with shape [N, C, H, W]
          global_features: Current estimate of globally smoothed image with shape [N, C, H, W]
          wedges: Tensor with shape [N, M, R, R, H', W'] containing rendered wedges
          patchsize:  Patchsize of each patch.
          stride: Patch stride.
          lmbda_wedge_mixing: Mixing parameter. Determines how much to weigh current
          junction parameters versus new parameter estimates when determining wedge colors.

        Returns:
          patches: Tensor of shape [N, C, R, R, H', W'] containing wedges with average feature superimposed
          wedge_colors: Tensor of shape [N, C, M, H', W'] with wedge average feature for each wedge of each patch
        """
        patchsize = self.patchsize if patchsize is None else patchsize
        stride = self.stride if stride is None else stride

        input_feature_patches = self.unfold(input_features, patchsize, stride)
        current_global_feature_patches = self.unfold(global_features, patchsize, stride)

        numerator = (torch.unsqueeze(input_feature_patches + lmbda_wedge_mixing * current_global_feature_patches,
                                     2) * torch.unsqueeze(wedges, 1)).sum([3, 4])
        denominator = (1.0 + lmbda_wedge_mixing) * torch.unsqueeze(wedges.sum([2, 3]), 1)

        wedge_colors = torch.unsqueeze(numerator / (denominator + 1e-10),3)
        wedge_colors = torch.unsqueeze(wedge_colors,3)
        patches = (torch.unsqueeze(wedges, 1) * wedge_colors).sum(dim=2)

        return patches, wedge_colors

    def dist2bdry(self, dist_boundaries: torch.Tensor, delta: float = None) -> torch.Tensor:
        """Convert a distance map into a boundary map."""

        delta = self.delta if delta is None else delta
        return 1 / (1 + (dist_boundaries / delta) ** 2)

    def make_square_patch_masks(self, rf_size: int, patchsize: int = None) -> torch.Tensor:
        """Make square patch masks."""
        patchsize = self.patchsize if patchsize is None else patchsize

        xy = torch.linspace(-torch.floor((patchsize - 1) / 2), torch.floor((patchsize - 1) / 2), patchsize)
        xlim, ylim = torch.meshgrid(xy, xy)
        mask = torch.where((torch.abs(xlim) < rf_size / 2) & (torch.abs(ylim) < rf_size / 2), 1, 0)

        return mask

    def make_circle_patch_masks(self, rf_size: int, patchsize: int = None) -> torch.Tensor:
        """Make circle patch masks."""
        patchsize = self.patchsize if patchsize is None else patchsize

        xy = torch.linspace(-torch.floor((patchsize - 1) / 2), torch.floor((patchsize - 1) / 2), patchsize)
        xlim, ylim = torch.meshgrid(xy, xy)
        mask = torch.where(torch.sqrt(xlim ** 2 + ylim ** 2) < rf_size / 2, 1, 0)

        return mask

    def get_scale_masks(self, scales: torch.Tensor, mask_shape: str = None, patchsize: int = None) -> torch.Tensor:
        """Get scale masks."""
        patchsize = self.patchsize if patchsize is None else patchsize
        mask_shape = self.mask_shape if mask_shape is None else mask_shape

        mask_list = []
        for scale in scales:
            if mask_shape == 'circle':
                mask_list.append(self.make_circle_patch_masks(scale, patchsize))
            elif mask_shape == 'square':
                mask_list.append(self.make_square_patch_masks(scale, patchsize))
            else:
                raise ValueError(f'Mask shape {mask_shape} not recognized.')

        masks = torch.stack(mask_list, dim=0)
        return masks

    def get_patch_density_and_masks(self,
                                    scales: torch.Tensor,
                                    mask_shape: str = None,
                                    patchsize: int = None,
                                    height: int = None,
                                    width: int = None):
        """Get patch density and masks."""
        mask_shape = self.mask_shape if mask_shape is None else mask_shape
        patchsize = self.patchsize if patchsize is None else patchsize
        height = self.height if height is None else height
        width = self.width if width is None else width

        masks = self.get_scale_masks(scales, mask_shape, patchsize)

        mask_density = torch.ones([1, 1, height, width])
        for mask in masks:
            unfolded_mask = self.unfold(mask, patchsize, self.stride)
            mask_density += self.fold(unfolded_mask, [1, 1, height, width], 'sum', self.stride)

        return mask_density, masks
    
    def get_alpha_omega_vertex(self, jparams, jparameterization=None, num_wedges=None):
        """Maps output of model to alpha, omega, vertex."""

        num_wedges = self.num_wedges if num_wedges is None else num_wedges
        jparameterization = self.jparameterization if jparameterization is None else jparameterization

        if jparameterization == 'standard':
            # default parameterization: (cos(alpha), sin(alpha), omega1, omega2,
            # omega3, u, v))
            # print('8888886666666688888888')
            # print(jparams.shape)
            alpha = torch.atan2(jparams[:, 1], jparams[:, 0]).unsqueeze(1)
            omega = jparams[:, 2:num_wedges+2]
            vertex = jparams[:, num_wedges+2:]


            # Normalize omega
            omega = omega * (2 * math.pi) / torch.sum(omega, dim=1, keepdim=True)
            # print('88888888999999999999988888888888')
            # print(alpha.shape)
            # print((omega/2).shape)
        else:
            raise NotImplementedError(f'{jparameterization} not a valid parameterization.')

        return alpha, omega, vertex

    def jparams2patches(self, jparams, jparameterization=None, num_wedges=None,
                        patchmin=None, patchmax=None, patchsize=None, delta=None,
                        eta=None):
        """Render boundary and wedge patches."""

        jparameterization = self.jparameterization if (jparameterization is
                                                       None) else jparameterization
        num_wedges = self.num_wedges if num_wedges is None else num_wedges
        patchmin = self.patchmin if patchmin is None else patchmin
        patchmax = self.patchmax if patchmax is None else patchmax
        patchsize = self.patchsize if patchsize is None else patchsize
        delta = self.delta if delta is None else delta
        eta = self.eta if eta is None else eta

        alpha, omega, vertex = self.get_alpha_omega_vertex(jparams,
                                                           jparameterization,
                                                           num_wedges)

        return self.get_local_maps(alpha, omega, vertex, patchmin, patchmax,
                                   patchsize, delta=delta, eta=eta)

    def get_local_maps(self, alpha, omega, vertex, patchmin=None, patchmax=None,
                   patchsize=None, delta=None, eta=None):
        """Render boundary and wedge patches."""

        patchmin = self.patchmin if patchmin is None else patchmin
        patchmax = self.patchmax if patchmax is None else patchmax
        patchsize = self.patchsize if patchsize is None else patchsize
        delta = self.delta if delta is None else delta
        eta = self.eta if eta is None else eta

        padding = (0,1,0,0)  # 修正为正确的元组格式
        # print('8888888888888888888')
        # print(alpha.shape)
        # print((omega/2).shape)
        # print('alpha shape:', alpha.shape)
        # print('omega shape:', omega.shape)
        # Compute wedge central angles
        # Compute wedge central angles
        # cumsum_omega = torch.cumsum(omega, dim=1)[:, :-1, ...]
        # padded_cumsum_omega = F.pad(cumsum_omega, padding)
        # # Ensure alpha and padded_cumsum_omega have matching shapes
        # if alpha.shape != padded_cumsum_omega.shape:
        #     # Adjust last dimension if needed
        #     if alpha.shape[-1] != padded_cumsum_omega.shape[-1]:
        #         diff = alpha.shape[-1] - padded_cumsum_omega.shape[-1]
        #         if diff > 0:
        #             padded_cumsum_omega = F.pad(padded_cumsum_omega, (0, diff, 0, 0))
        #         else:
        #             alpha = F.pad(alpha, (0, -diff, 0, 0))
            
        #     # Adjust second dimension if needed
        #     if alpha.shape[1] != padded_cumsum_omega.shape[1]:
        #         diff = alpha.shape[1] - padded_cumsum_omega.shape[1]
        #         if diff > 0:
        #             padded_cumsum_omega = F.pad(padded_cumsum_omega, (0, 0, 0, 0, 0, diff))
        #         else:
        #             alpha = F.pad(alpha, (0, 0, 0, 0, 0, -diff))
        # print('alpha shape:', alpha.shape)
        # print('omega / 2 shape:', (omega / 2).shape)
        # print('padded_cumsum_omega shape:', padded_cumsum_omega.shape)
        # centralangles = torch.unsqueeze(alpha + omega / 2 + padded_cumsum_omega, 2)
        # centralangles = torch.unsqueeze(centralangles, 3)
        centralangles = torch.unsqueeze(alpha + omega / 2 + omega, dim=2)
        centralangles = torch.unsqueeze(centralangles, dim=2)
        # Compute wedge angles
        wedgeangles = torch.unsqueeze(omega * (2 * math.pi) / omega, dim = 2)
        wedgeangles = torch.unsqueeze(wedgeangles, dim=2)
        # print(centralangles.shape)
        # print(wedgeangles.shape)
        # Compute wedge boundary angles
        boundaryangles = torch.unsqueeze(alpha + omega, dim=2)
        boundaryangles = torch.unsqueeze(boundaryangles, dim=2)
        # Render and return boundary and feature patches
        feature_patches = self.render_wedges(vertex, centralangles, wedgeangles,
                                            patchmin, patchmax, patchsize, eta)
        distance_patches = self.render_distance(vertex, boundaryangles, patchmin,
                                                patchmax, patchsize)
        boundary_patches = self.render_boundaries(vertex, boundaryangles, patchmin,
                                                patchmax, patchsize, delta)

        return feature_patches, distance_patches, boundary_patches
