import numpy as np
import torch


class ERA5Converter():
    """Module used to convert ERA5 data to spherical coordinates and features.

    Args:
        data_shape (tuple of ints): Tuple of the form (num_lats, num_lons).
        normalize (bool): This argument is only kept for compatibility.
            Coordinates will always lie in [-1, 1] since we use spherical
            coordinates with r=1.
        normalize_features (bool): If True normalizes features (e.g. temperature
            values) to lie in [-1, 1]. This assumes features from the dataloader
            lie in [0, 1].

    Notes:
        We assume the spherical data is given as a tensor of shape
        (3, num_lats, num_longs), where the first dimension contains latitude
        values, the second dimension longitude values and the third dimension
        temperature values.

        The coordinates are given by:
            x = cos(latitude) cos(longitude)
            y = cos(latitude) sin(longitude)
            z = sin(latitude).
    """

    def __init__(self, device, data_shape, normalize=True,
                 normalize_features=False, resolution=1):
        self.device = device
        self.data_shape = data_shape
        self.normalize = normalize
        self.normalize_features = normalize_features
        # Initialize coordinates
        self.latitude = np.linspace(90., -90., data_shape[0]*resolution)
        self.longitude = np.linspace(0., 360. - (360. / (data_shape[1]*resolution)),
                                     data_shape[1]*resolution)
        # Create a grid of latitude and longitude values (num_lats, num_lons)
        longitude_grid, latitude_grid = np.meshgrid(self.longitude,
                                                    self.latitude)
        # Shape (3, num_lats, num_lons) (add bogus temperature dimension to be
        # compatible with coordinates and features transformation function)
        data_tensor = np.stack([latitude_grid,
                                longitude_grid,
                                np.zeros_like(longitude_grid)])
        data_tensor = torch.Tensor(data_tensor).to(device)
        # Shape (num_lats, num_lons, 3)
        self.coordinates, _ = era5_to_coordinates_and_features(data_tensor)
        # (num_lats, num_lons, 3) -> (num_lats * num_lons, 3)
        self.coordinates = self.coordinates.view(-1, 3)
        # Store to use when converting to from coordinates and features to data
        self.latitude_grid = torch.Tensor(latitude_grid).to(device)
        self.longitude_grid = torch.Tensor(longitude_grid).to(device)

    def to_coordinates_and_features(self, data, resolution=1):
        """Given a datapoint convert to coordinates and features at each
        coordinate.

        Args:
            data (torch.Tensor): Shape (3, num_lats, num_lons) where latitudes
                and longitudes are in degrees and temperatures are in [0, 1].
        """
        # Shapes (num_lats, num_lons, 3), (num_lats, num_lons, 1)
        coordinates, features = era5_to_coordinates_and_features(data, resolution)
        if self.normalize_features:
            # Features are in [0, 1], convert to [-1, 1]
            features = 2. * features - 1.
        # Flatten features and coordinates
        # (num_lats, num_lons, 1) -> (num_lats * num_lons, 1)
        features = features.view(-1, 1)
        # (num_lats, num_lons, 3) -> (num_lats * num_lons, 3)
        coordinates = coordinates.view(-1, 3)
        return coordinates, features

    def batch_to_coordinates_and_features(self, data_batch, resolution=1):
        """Given a batch of datapoints, convert to coordinates and features at
        each coordinate.

        Args:
            data_batch (torch.Tensor): Shape (batch_size, 3, num_lats, num_lons)
                where latitudes and longitudes are in degrees and temperatures
                are in [0, 1].
        """
        batch_size = data_batch.shape[0]
        # Shapes (batch_size, num_lats, num_lons, 3), (batch_size, num_lats, num_lons, 1)
        coordinates_batch, features_batch = era5_to_coordinates_and_features(data_batch, resolution=resolution)
        if self.normalize_features:
            # Image features are in [0, 1], convert to [-1, 1]
            features_batch = 2. * features_batch - 1.
        # Flatten features and coordinates
        # (batch_size, num_lats, num_lons, 1) -> (batch_size, num_lats * num_lons, 1)
        features_batch = features_batch.view(batch_size, -1, 1)
        # (batch_size, num_lats, num_lons, 3) -> (batch_size, num_lats * num_lons, 3)
        coordinates_batch = coordinates_batch.view(batch_size, -1, 3)
        return coordinates_batch, features_batch

    def to_data(self, coordinates, features, resolution=1):
        """Converts tensors of features and coordinates to ERA5 data.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (num_lats * num_lons, 1).
            resolution (int): Unused argument.

        Notes:
            Since we don't use subsampling or superresolution for ERA5
            data, this function ignores passed coordinates tensor and
            assumes we use self.coordinates.
        """
        if self.normalize_features:
            # [-1, 1] -> [0, 1] (keep masks still zeros)
            features[features == 0] = -1
            features = .5 * (features + 1.)
        # Reshape features (num_lats * num_lons, 1) -> (1, num_lats, num_lons)
        features = features.view(1, self.data_shape[0] * resolution, self.data_shape[1] * resolution)
        # Shape (3, num_lats, num_lons)
        return torch.cat([self.latitude_grid.unsqueeze(0),
                          self.longitude_grid.unsqueeze(0),
                          features], dim=0)

    def batch_to_data(self, coordinates, features, resolution=1):
        """Converts tensor of batch features to point cloud representation.

        Args:
            coordinates (torch.Tensor): Unused argument.
            features (torch.Tensor): Shape (batch_size, num_lats, num_lons, 1).
        """
        batch_size = features.shape[0]
        if self.normalize_features:
            # [-1, 1] -> [0, 1]
            features = .5 * (features + 1.)
        # Reshape features (batch_size, num_lats * num_lons, 1) -> (batch_size, 1, num_lats, num_lons)
        features = features.view(batch_size, 1, self.data_shape[0] * resolution, self.data_shape[1] * resolution)
        # Shape (batch_size, 1, num_lats, num_lons)
        batch_lat_grid = self.latitude_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        batch_lon_grid = self.longitude_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Shape (batch_size, 3, num_lats, num_lons)
        return torch.cat([batch_lat_grid, batch_lon_grid, features], dim=1)

    def unnormalized_coordinates(self, coordinates):
        """
        """
        unnormalized_coordinates = coordinates / 2 + 0.5
        return unnormalized_coordinates * (self.data_shape[1] - 1)

    def superresolve_coordinates(self, resolution):
        """Not implemented for spherical data."""
        raise NotImplementedError


def era5_to_coordinates_and_features(data, use_spherical=True, resolution=1):
    """
    Converts ERA5 data lying on the globe to spherical coordinates and features.
    The coordinates are given by:
        x = cos(latitude) cos(longitude)
        y = cos(latitude) sin(longitude)
        z = sin(latitude).
    The features are temperatures.

    Args:
        data (torch.Tensor): Tensor of shape ({batch,} 3, num_lats, num_lons)
            as returned by the ERA5 dataloader (batch dimension optional).
            The first dimension contains latitudes, the second longitudes
            and the third temperatures.
        use_spherical (bool): If True, uses spherical coordinates, otherwise
            uses normalized latitude and longitude directly.

    Returns:
        Tuple of coordinates and features where coordinates has shape
        ({batch,} num_lats, num_lons, 2 or 3) and features has shape
        ({batch,} num_lats, num_lons, 1).
    """
    assert data.ndim in (3, 4)

    if data.ndim == 3:
        latitude, longitude, temperature = data
    elif data.ndim == 4:
        latitude, longitude, temperature = data[:, 0], data[:, 1], data[:, 2]

    if resolution != 1:
        latitude = torch.nn.functional.interpolate(latitude.unsqueeze(1), scale_factor=resolution, mode='bilinear').squeeze()
        longitude = torch.nn.functional.interpolate(longitude.unsqueeze(1), scale_factor=resolution, mode='bilinear').squeeze()
        temperature = torch.nn.functional.interpolate(temperature.unsqueeze(1), scale_factor=resolution, mode='bilinear').squeeze()

    # Create coordinate tensor
    if use_spherical:
        coordinates = torch.zeros(latitude.shape + (3,)).to(latitude.device)
        long_rad = deg_to_rad(longitude)
        lat_rad = deg_to_rad(latitude)
        coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
        coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
        coordinates[..., 2] = torch.sin(lat_rad)
    else:
        coordinates = torch.zeros(latitude.shape + (2,)).to(latitude.device)
        # Longitude [0, 360] -> [-1, 1]
        coordinates[..., 0] = longitude / 180. - 1.
        # Latitude [-90, 90] -> [-.5, .5]
        coordinates[..., 1] = latitude / 180.
    # Feature tensor is given by temperatures (unsqueeze to ensure we have
    # feature dimension)
    features = temperature.unsqueeze(-1)

    return coordinates, features


def deg_to_rad(degrees):
    return np.pi * degrees / 180.


def rad_to_deg(radians):
    return 180. * radians / np.pi


def normalize_coordinates(coordinates, max_coordinate):
    """Normalizes coordinates to [-1, 1] range.

    Args:
        coordinates (torch.Tensor):
        max_coordinate (float): Maximum coordinate in original grid.
    """
    # Get points in range [-0.5, 0.5]
    normalized_coordinates = coordinates / (max_coordinate - 1) - 0.5
    # Convert to range [-1, 1]
    normalized_coordinates *= 2
    return normalized_coordinates
