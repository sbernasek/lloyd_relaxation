import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import Voronoi
from animation import RelaxationAnimation


class LloydRelaxation:
    """
    Class enables repeated Lloyd relaxation of a single set of xy coordinates.

    Attributes:
    xycoords_initial (np array) - initial coordinates, retained for comparison
    xycoords (np array) - coordinates after relaxation
    bounding_box (array like) - values defining rectangular region, i.e. [xmin, xmax, ymin, ymax]
    boundary (np array) - xy data, including all reflections
    voronoi (scipy.spatial.Voronoi object) - filtered regions
    """

    def __init__(self, xycoords):
        """
        Args:
        xycoords (np array) - xy data
        """
        self.xycoords_initial = xycoords
        self.xycoords = xycoords
        self.bounding_box, self.boundary, self.voronoi = None, None, None
        self.update_bounding_box()
        self.update_boundary()
        self.update_voronoi()
        self.history = []

    def reset(self):
        """ Revert to initial values. """
        self.__init__(deepcopy(self.xycoords_initial))

    @staticmethod
    def _get_bounding_box(xycoords, pad=0.01):
        """
        Get bounds of rectangular region encompassing xy data.

        Args:
        xycoords (np array) - xy data
        pad (float) - translational separation inserted between data and reflection if no bounding_box is provided

        Returns:
        bounding_box (array like) - values defining rectangular region, i.e. [xmin, xmax, ymin, ymax]
        """
        xmin, ymin = xycoords.min(axis=1)
        xmax, ymax = xycoords.max(axis=1)
        bounding_box = np.array([xmin-pad, xmax+pad, ymin-pad, ymax+pad])
        return bounding_box

    def update_bounding_box(self):
        """ Update bounding box. """
        self.bounding_box = self._get_bounding_box(self.xycoords)

    @staticmethod
    def _get_boundary(xycoords, bounding_box):
        """
        Generate rectangular bounded voronoi regions by reflecting all data about each edge of bounding_box.

        Args:
        xycoords (np array) - xy data
        bounding_box (array like) - values defining rectangular region, i.e. [xmin, xmax, ymin, ymax]

        Returns:
        boundary (np array) - xy data, including all reflections
        """

        # reflect about left boundary
        points_left = np.copy(xycoords.T)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])

        # reflect about right boundary
        points_right = np.copy(xycoords.T)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])

        # reflect about bottom boundary
        points_down = np.copy(xycoords.T)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])

        # reflect about top boundary
        points_up = np.copy(xycoords.T)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

        boundary = np.hstack((points_left.T, points_right.T, points_down.T, points_up.T))

        return boundary

    def update_boundary(self):
        """ Update boundary by reflecting xy-coordinates across the bounding_box edges. """
        self.boundary = self._get_boundary(self.xycoords, self.bounding_box)

    def get_voronoi(self, xycoords, boundary, bounding_box):
        """
        Construct voronoi regions and filter those outside the bounding_box.

        Args:
        xycoords (np array) - xy data
        boundary (np array) - xy data, including all reflections
        bounding_box (array like) - values defining rectangular region, i.e. [xmin, xmax, ymin, ymax]

        Returns:
        voronoi (scipy.spatial.Voronoi object) - filtered regions
        """

        # combine coordinates and boundary points
        points = np.hstack((xycoords, boundary))

        # construct voronoi regions
        voronoi = Voronoi(points.T)

        # filter regions outside of bounding box
        voronoi.filtered_points = voronoi.points[0:xycoords.shape[1], :]
        voronoi.filtered_region_indices, voronoi.filtered_regions = self.filter_regions(voronoi, bounding_box)

        return voronoi

    def update_voronoi(self):
        """ Get region-filtered Voronoi object. """
        self.voronoi = self.get_voronoi(self.xycoords, self.boundary, self.bounding_box)

    @staticmethod
    def filter_regions(voronoi, bounding_box):
        """
        Filter voronoi regions such that only those within the bounding_box are retained.

        Args:
        voronoi (scipy.spatial.Voronoi object)
        bounding_box (array like) - values defining rectangular region, i.e. [xmin, xmax, ymin, ymax]

        Returns:
        indices (list) - indices of retained regions
        regions (list of lists) - list of vertex indices for each retained region
        """
        eps = 1e-5
        indices, regions = [], []
        for i, region in enumerate(voronoi.regions):
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = voronoi.vertices[index, 0]
                    y = voronoi.vertices[index, 1]
                    if not(bounding_box[0] - eps <= x
                           and x <= bounding_box[1] + eps
                           and bounding_box[2] - eps <= y
                           and y <= bounding_box[3] + eps):
                        flag = False
                        break
            if region != [] and flag:
                indices.append(i)
                regions.append(region)
        return indices, regions

    @staticmethod
    def _get_centroids(voronoi):
        """
        Get centroids for all voronoi regions within the bounding box.

        Args:
        voronoi (scipy.spatial.Voronoi object)

        Returns:
        centroids (np array)
        """

        centroids = []
        for region in voronoi.filtered_regions:
            region_vertices = voronoi.vertices[region + [region[0]], :]
            centroid = get_centroid_of_region(region_vertices)
            centroids.append(list(centroid[0, :]))

        # order like existing points
        region_to_filtered = {region_index: filtered_index for filtered_index, region_index in enumerate(voronoi.filtered_region_indices)}
        point_indices = [region_to_filtered[region] for region in voronoi.point_region[0: len(centroids)]]
        return np.array([centroids[point_index] for point_index in point_indices])

    def get_centroids(self):
        """ Returns centroids of voronoi regions. """
        return self._get_centroids(self.voronoi)

    def iterate(self):
        """
        Execute single iteration of LLoyd relaxation. Coordinates are merged with boundary, voronoi regions are
        constructed and filtered such that only those inside the bounding box remain. The centroids of retained regions
        are returned.

        Returns:
        centroids (np array) - centroids of filtered voronoi cells, ordered by position in xycoords
        """
        self.update_boundary()
        self.update_voronoi()
        centroids = self.get_centroids()
        return centroids.T

    def run(self, n_iters=100, record=False):
        """
        Run Lloyd's relaxation.

        Args:
        n_iters (int) - iterations performed
        record (bool) - if True, append each voronoi cell to history
        """
        self.reset()
        for _ in range(n_iters):
            self.xycoords = self.iterate()

            # save current voronoi cell
            if record:
                self.history.append(deepcopy(self.voronoi))

    @staticmethod
    def _plot_xy(voronoi, ax, point_color='blue', point_alpha=0.5):
        artist = ax.plot(voronoi.filtered_points[:, 0], voronoi.filtered_points[:, 1], '.', color=point_color, alpha=point_alpha)
        return artist

    @staticmethod
    def _plot_ridge_vertices(voronoi, ax, vertex_color='black', vertex_alpha=0.5):
        artists = []
        for region in voronoi.filtered_regions:
            vertices = voronoi.vertices[region, :]
            artist = ax.scatter(vertices[:, 0], vertices[:, 1], c=vertex_color, s=1, alpha=vertex_alpha)
            artists.append(artist)
        return artists

    @staticmethod
    def _plot_ridges(voronoi, ax, ridge_color='black', ridge_alpha=0.5):
        artists = []
        for region in voronoi.filtered_regions:
            vertices = voronoi.vertices[region + [region[0]], :]
            artist = ax.plot(vertices[:, 0], vertices[:, 1], '-', color=ridge_color, alpha=ridge_alpha)
            artists.append(artist)
        return artists


    @staticmethod
    def _plot_voronoi_regions(voronoi, ax=None,
                  point_color='blue', ridge_color='black', vertex_color='black',
                  point_alpha=0.5, ridge_alpha=0.5, vertex_alpha=0.0):
        """
        Plot voronoi vertices and ridges.

        Args:
        voronoi (scipy.spatial.Voronoi object)
        ax (axes object) - if None, figure is created
        point_color, ridge_color, vertex_color (str)
        point_alpha, ridge_alpha, vertex_alpha (float)

        Returns:
        point_artists (matplotlib artists)
        """

        # if no axes provided, create figure
        if ax is None:
            fig, ax = plt.subplots()

        # Plot xy data
        point_artists = LloydRelaxation._plot_xy(voronoi, ax, point_color=point_color, point_alpha=point_alpha)

        # Plot ridge vertices
        vertex_artists = LloydRelaxation._plot_ridge_vertices(voronoi, ax, vertex_color=vertex_color, vertex_alpha=vertex_alpha)

        # Plot ridges
        ridge_artists = LloydRelaxation._plot_ridges(voronoi, ax, ridge_color=ridge_color, ridge_alpha=ridge_alpha)

        xmin, ymin = voronoi.filtered_points.min(axis=0)
        xmax, ymax = voronoi.filtered_points.max(axis=0)
        ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
        return point_artists

    def plot_translation_field(self, **kwargs):
        """
        Plot vector field of line-segments between initial and relaxed xy coordinates.

        kwargs: quiver plot formatting arguments
        """
        plot_vector_field(self.xycoords_initial, self.xycoords, **kwargs)

    def plot_voronoi_regions(self, **kwargs):
        """
        Plot bounded voronoi regions.

        kwargs: plot formatting arguments
        """
        artists = self._plot_voronoi_regions(self.voronoi, **kwargs)

    def get_video(self, **kwargs):
        """
        Get HTML5 video of relaxation.

        kwargs: video formatting arguments
        """
        return RelaxationAnimation(self).get_video(**kwargs)


# additional utility functions

def get_centroid_of_region(vertices):
    """
    Get centroid of a set of vertices.

    Args:
    vertices (np array) - voronoi vertices for a single region

    Returns:
    centroid (np array)
    """

    # Polygon's signed area, centroid's x and y
    A, C_x, C_y = 0, 0, 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A += s
        C_x += (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y += (vertices[i, 1] + vertices[i + 1, 1]) * s
    A *= 0.5
    C_x *= (1.0 / (6.0 * A))
    C_y *= (1.0 / (6.0 * A))
    return np.array([[C_x, C_y]])


def plot_vector_field(before, after, **kwargs):
    """
    Plot vector field of line-segments between two ordered arrays of xy coordinates.

    Args:
    before, after (np array) - xy coordinates
    kwargs: quiver formatting arguments

    Returns:
    ax (axes object)
    """
    # create figure
    fig, ax = plt.subplots()

    # compute differences
    x, y = before
    u, v = after - before

    # plot vector field
    ax.quiver(x, y, u, v, units='xy', scale=1, **kwargs)

    # set aspect ratio and plot limits
    _ = plt.axis('equal')
    xmin, ymin = np.hstack((before, after)).min(axis=1)
    xmax, ymax = np.hstack((before, after)).max(axis=1)
    ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
    return ax
