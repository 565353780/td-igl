from td_igl.Dataset.shapenet import ShapeNet
from td_igl.Dataset.axis_scaling import AxisScaling


def build_shape_surface_occupancy_dataset(split, args):
    if split == "train":
        transform = AxisScaling((0.75, 1.25), True)
        return ShapeNet(
            args.data_path,
            split=split,
            transform=transform,
            sampling=True,
            num_samples=1024,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )
    elif split == "val":
        return ShapeNet(
            args.data_path,
            split=split,
            transform=None,
            sampling=False,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )
    else:
        return ShapeNet(
            args.data_path,
            split=split,
            transform=None,
            sampling=False,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )
