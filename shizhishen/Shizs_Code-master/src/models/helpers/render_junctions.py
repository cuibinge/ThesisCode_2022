import torch
import torch.nn.functional as F
import math
class JunctionRenderer:
    """Renders the junctions of the wedge support and distance maps."""

    def render_wedges(self,
                      vertex: torch.Tensor,
                      centralangles: torch.Tensor,
                      wedgeangles: torch.Tensor,
                      patchmin: float,
                      patchmax: float,
                      patchres: int,
                      eta: float) -> torch.Tensor:
        """Render an integer-valued image of the wedge supports over a square patch.

        Args:
          vertex: Tensor of shape [N, 2, H, W] containing the u and v coordinates
          of vertices
          centralangles: Tensor of shape [N, 3, H, W] containing the three central
          angles (wedge directions)
          wedgeangles: Tensor of shape [N, 3, H, W] containing the three wedge angles
          that sum to 2*pi
          patchmin: Minimum value of the patch
          patchmax: Maximum value of the patch
          patchres: Size of the patch in pixels
          eta: Tensor of shape [N, 1] containing the angular speed of the wedge
          support

        Returns:
          Tensor of shape [N, M, R, R, H, W]
        """
        # coordinate grid of pixel locations
        yt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, patchres, 1, 1, 1).expand(-1, -1, -1, patchres, -1, -1).cuda()
        xt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, 1, patchres, 1, 1).expand(-1, -1, patchres, -1, -1, -1).cuda()

        x0 = vertex[:, 0, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        y0 = vertex[:, 1, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()

        cos_ca = torch.cos(centralangles).cuda()
        sin_ca = torch.sin(centralangles).cuda()
      # # Print and inspect tensor values
      #   print("yt:", yt)
      #   print("xt:", xt)
      #   print("x0:", x0)
      #   print("y0:", y0)
      #   print("cos_ca:", cos_ca)
      #   print("sin_ca:", sin_ca)
        x = ((xt - x0) * cos_ca +
             (yt - y0) * sin_ca  -
             torch.cos(wedgeangles / 2) * torch.sqrt((xt - x0)**2 + (yt - y0)**2))

        x = 0.5 * (1.0 + (2.0 / math.pi) * torch.atan(x / eta))
        x = x / torch.sum(x, dim=1, keepdim=True)

        return x

    def render_distance(self,
                        vertex: torch.Tensor,
                        boundaryangles: torch.Tensor,
                        patchmin: float,
                        patchmax: float,
                        patchres: int) -> torch.Tensor:
        """Render a distance map over a square patch.

        Args:
          vertex: Tensor of shape [N, 2, H, W] containing the u and v coordinates
          of the vertices
          boundaryangles: Tensor of shape [N, 3, H, W] containing the three boundary
          angles (boundary-ray directions)
          patchmin: Minimum value of the patch
          patchmax: Maximum value of the patch
          patchres: Size of the patch

        Returns:
          Tensor of shape [N, 1, R, R, H, W]
        """
        # coordinate grid of pixel locations
        yt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, patchres, 1, 1, 1).expand(-1, -1, -1, patchres, -1, -1).cuda()
        xt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, 1, patchres, 1, 1).expand(-1, -1, patchres, -1, -1, -1).cuda()

        x0 = vertex[:, 0, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        y0 = vertex[:, 1, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()

        cos_ba = torch.cos(boundaryangles).cuda()
        sin_ba = torch.sin(boundaryangles).cuda()
    # Print and inspect tensor values
        # print("yt:", yt.detach().cpu().numpy())
        # print("xt:", xt.detach().cpu().numpy())
        # print("x0:", x0.detach().cpu().numpy())
        # print("y0:", y0.detach().cpu().numpy())
        # print("cos_ca:", cos_ca.detach().cpu().numpy())
        # print("sin_ca:", sin_ca.detach().cpu().numpy())
        distance_branches = torch.where(
            0 < ((xt - x0) * cos_ba + (yt - y0) * sin_ba),
            torch.abs(-(xt - x0) * sin_ba + (yt - y0) * cos_ba),
            torch.sqrt((xt - x0)**2 + (yt - y0)**2)
        )

        # final distance is minimum over arms, expanded to [N, 1, R, R, H, W]
        distance = torch.min(distance_branches, dim=1, keepdim=True)[0] * (patchres / (patchmax - patchmin))

        return distance

    def render_boundaries(self,
                          vertex: torch.Tensor,
                          boundaryangles: torch.Tensor,
                          patchmin: float,
                          patchmax: float,
                          patchres: int,
                          delta: float = 0.005) -> torch.Tensor:
        """Render an image of the wedge boundaries over a square patch.

        Args:
          vertex: Tensor of shape [N, 2, H, W] containing the u and v coordinates
          of the vertices
          boundaryangles: Tensor of shape [N, 3, H, W] containing the three boundary
          angles (boundary-ray directions)
          patchmin: Minimum value of the patch
          patchmax: Maximum value of the patch
          patchres: Size of the patch
          delta: Delta value of the patch

        Returns:
          Tensor of shape [N, 1, R, R, H, W]
        """
        # coordinate grid of pixel locations
        yt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, patchres, 1, 1, 1).expand(-1, -1, -1, patchres, -1, -1).cuda()
        xt = torch.linspace(patchmin, patchmax, patchres).view(1, 1, 1, patchres, 1, 1).expand(-1, -1, patchres, -1, -1, -1).cuda()

        x0 = vertex[:, 0, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        y0 = vertex[:, 1, ...].unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()

        cos_ba = torch.cos(boundaryangles).cuda()
        sin_ba = torch.sin(boundaryangles).cuda()

        # Use [1 / (1 + (x/opts.delta)**2 )] for the relaxed dirac distribution
        x = ((xt - x0) * cos_ba + (yt - y0) * sin_ba -
             torch.sqrt((xt - x0)**2 + (yt - y0)**2))
        r = torch.sqrt((xt - x0)**2 + (yt - y0)**2)

        patches = 1.0 / (1.0 + ((x * r) / delta)**2)

        standard_boundaries = torch.max(patches, dim=1, keepdim=True)[0]

        return standard_boundaries
