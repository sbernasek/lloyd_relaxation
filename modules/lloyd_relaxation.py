import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from operator import mul
from functools import reduce
from scipy.spatial import Voronoi
from matplotlib.path import Path
from animation import RelaxationAnimation


class LloydRelaxation:
    """
    Class enables repeated Lloyd relaxation of a single set of xy coordinates.

    Attributes:
    xycoords_initial (np array) - initial coordinates, retained for comparison
    xycoords (np array) - coordinates after relaxation
    boundary (np array) - bounding vertices of relaxable region, 2 x M
    bounding_points (np array) - reflections of xy data about all boundaries
    voronoi (scipy.spatial.Voronoi object) - filtered regions
    """

    def __init__(self, xycoords, **kwargs):
        """
        Args:
        xycoords (np array) - xy data
        boundary_type (str) - type of boundary used, e.g. 'box' or None
        kwargs: boundary padding arguments
        """
        self.xycoords_initial = xycoords
        self.xycoords = xycoords
        self.boundary = None
        self.bounding_points = None
        self.voronoi = None
        self.init_params = kwargs
        self.set_boundary(**kwargs)
        self.update_bounding_points()
        self.update_voronoi()
        self.history = []

    def reset(self):
        """ Revert to initial values. """
        self.__init__(deepcopy(self.xycoords_initial), **self.init_params)

    @staticmethod
    def get_intervertex_vectors(vertices):
        """
        Return vector between sequential clockwise-ordered vertices.

        Args:
        vertices (np array) - M vertices between which vectors are constructed, 2 x M

        Returns:
        line_vectors (np array) - vector lying on each line, 2 x M
        """
        line_vectors = np.diff(np.append(vertices, vertices[:, 0].reshape(2, 1), axis=1))
        return line_vectors

    @staticmethod
    def get_reflection(xycoords, line_points, line_vectors):
        """
        Returns reflection of N points about M lines.

        Args:
        xycoords (np array) - N points to be reflected, 2 x N
        line_points (np array) - point on each of M lines, 2 x M
        line_vectors (np array) - vector on each of M lines, 2 x M

        Returns:
        reflection (np array) - flattened reflection of N points about M lines, 2 x (N x M)
        """

        # construct tiled vectors
        x = np.tile(xycoords.reshape(xycoords.shape[0], xycoords.shape[1], 1), (1, 1, line_points.shape[1]))
        p = np.tile(line_points.reshape(line_points.shape[0], 1, line_points.shape[1]), (1, xycoords.shape[1], 1))
        d = np.tile(line_vectors.reshape(line_vectors.shape[0], 1, line_vectors.shape[1]), (1, xycoords.shape[1], 1))

        # instantiate dot operation
        dot = lambda a, b: np.einsum("ijk, ijk->jk", a, b)

        # compute projections
        projection_magnitudes = dot(x-p, d) / dot(d, d)
        projection_onto_lines = p + d*np.tile(projection_magnitudes.reshape(1, projection_magnitudes.shape[0], projection_magnitudes.shape[1]), reps=(2, 1, 1))

        #compute reflections
        reflection = 2*projection_onto_lines - x

        return reflection.reshape(reflection.shape[0], -1)

    @staticmethod
    def sort_clockwise(xycoords):
        """ Returns clockwise-sorted xy coordinates. """
        return xycoords[:, np.argsort(np.arctan2(*(xycoords.T - xycoords.mean(axis=1)).T))]

    @staticmethod
    def find_edge_regions(voronoi):
        """ Check whether each region is a border region. """
        return np.array(list(map(lambda x: -1 in x, np.take(voronoi.regions, voronoi.point_region))))

    @classmethod
    def _get_edges(cls, xycoords):
        """
        Get array of points on edge of voronoi region constructed about xycoords.

        Args:
        xycoords (np array) - xy data, 2 x N

        Returns:
        edges (np array) - 2 x M
        """
        edges = cls.sort_clockwise(xycoords[:, cls.find_edge_regions(cls._get_voronoi(xycoords))])
        return edges

    @staticmethod
    def _get_bounding_box(xycoords):
        """
        Get edges of rectangular region encompassing xy data.

        Args:
        xycoords (np array) - xy data

        Returns:
        boundary (np array) - bounding vertices of relaxable region, 2 x M
        """
        xmin, ymin = xycoords.min(axis=1)
        xmax, ymax = xycoords.max(axis=1)
        edges = np.array([[xmin, xmin, xmax, xmax], [ymin, ymax, ymax, ymin]])
        return edges

    @staticmethod
    def _get_bounding_circle(xycoords):
        """ Get edges of circular region encompassing xy data. """
        center = xycoords.mean(axis=1).reshape(2, 1)
        radius = np.sqrt(((xycoords-center)**2).sum(axis=0)).max()
        theta = np.arange(0, 1.95*np.pi, 0.1)[::-1]
        edges = np.vstack((radius*np.cos(theta), radius*np.sin(theta))) + center
        return edges

    @classmethod
    def _get_boundary(cls, edges, dilation=1.01):
        """
        Construct boundary from points on edge of dataset, dilated away from centroid.

        Args:
        edges (np array) - 2 x M

        Returns:
        boundary (np array) - bounding vertices of relaxable region, 2 x M
        """
        centroid = cls.get_centroid_of_region(np.append(edges, edges[:, 0].reshape(2, 1), axis=1).T)
        boundary = np.apply_along_axis(lambda x: centroid + dilation*(x-centroid), axis=0, arr=edges)
        return boundary

    @classmethod
    def get_edges(cls, xycoords, boundary_type=None):
        """ Get edges for boundary. """

        # rectangular
        if boundary_type in ('box', 'rectangular', 'square'):
            return cls._get_bounding_box(xycoords)

        # circular
        elif boundary_type in ('circle', 'circular', 'round'):
            return cls._get_bounding_circle(xycoords)

        # empirical
        else:
            return cls._get_edges(xycoords)

    def set_boundary(self, boundary_type=None, dilation=1.001):
        """ Set boundary. """
        edges = self.get_edges(self.xycoords, boundary_type=boundary_type)
        self.boundary = self._get_boundary(edges=edges, dilation=dilation)

    def get_boundary_limits(self):
        """ Get lower and upper limits for boundary. """
        xmin, ymin = self.boundary.min(axis=1)
        xmax, ymax = self.boundary.max(axis=1)
        return (xmin, xmax), (ymin, ymax)

    @classmethod
    def _get_bounding_points(cls, xycoords, boundary):
        """
        Construct periodic boundary conditions by reflecting data about each edge of boundary.

        Args:
        xycoords (np array) - xy data
        boundary (np array) - bounding vertices of relaxable region, 2 x M

        Returns:
        bounding_points (np array) - reflections of xy data about all boundaries, 2 x (N x M)
        """
        boundary_vectors = cls.get_intervertex_vectors(boundary)
        bounding_points = cls.get_reflection(xycoords, boundary, boundary_vectors)
        return bounding_points

    def update_bounding_points(self):
        """ Update periodic boundary conditions. """
        self.bounding_points = self._get_bounding_points(self.xycoords, self.boundary)

    @staticmethod
    def _get_voronoi(xycoords):
        return Voronoi(xycoords.T)

    @classmethod
    def get_voronoi(cls, xycoords, bounding_points, boundary):
        """
        Construct voronoi regions and filter those outside the boundary.

        Args:
        xycoords (np array) - xy data
        bounding_points (np array) - reflections of xy data about all boundaries, 2 x (N x M)
        boundary (np array) - bounding vertices of relaxable region, 2 x M

        Returns:
        voronoi (scipy.spatial.Voronoi object) - filtered regions
        """

        # combine coordinates and boundary points
        points = np.hstack((xycoords, bounding_points))

        # construct voronoi regions
        voronoi = cls._get_voronoi(points)

        # filter regions outside of bounding box
        voronoi.filtered_points = cls.filter_points(points, boundary)
        voronoi.filtered_region_indices, voronoi.filtered_regions = cls.filter_regions(voronoi, boundary)

        return voronoi

    def update_voronoi(self):
        """ Get region-filtered voronoi object. """
        self.voronoi = self.get_voronoi(self.xycoords, self.bounding_points, self.boundary)

    @staticmethod
    def get_vertex_mask(xycoords, vertices):
        """
        Determines which of N points fall within region enclosed by M vertices.

        Args:
        xycoords (np array) - points to be checked, 2 x N
        vertices (np array) - vertices, 2 x M

        Returns:
        within (np array) - boolean mask, if True point is within region
        """

        path = Path(vertices.T, closed=False)
        within = path.contains_points(xycoords.T, radius=-1e-10)
        return within

    @classmethod
    def filter_points(cls, xycoords, boundary):
        """
        Filter xy coordinates such that only those within the boundary are retained.

        Args:
        xycoords (np array) - input points, 2 x N
        boundary (np array) - bounding vertices of relaxable region, 2 x M

        Returns:
        indices (list) - indices of retained points
        points (list of lists) - retained points
        """

        mask = cls.get_vertex_mask(xycoords, boundary)
        return xycoords[:, mask]

    @classmethod
    def filter_regions(cls, voronoi, boundary):
        """
        Filter voronoi regions such that only those within the boundary are retained.

        Args:
        voronoi (scipy.spatial.Voronoi object)
        boundary (np array) - bounding vertices of relaxable region, 2 x M

        Returns:
        indices (list) - indices of retained regions
        regions (list of lists) - list of vertex indices for each retained region
        """

        # determine whether each voronoi vertex is within the boundary
        mask = cls.get_vertex_mask(voronoi.vertices.T, boundary)

        indices, regions = zip(*[(i, region) for i, region in enumerate(voronoi.regions)
                                 if reduce(mul, mask[region], True) == True
                                 and region != []
                                 and -1 not in region])

        return indices, regions

    @staticmethod
    def get_centroid_of_region(vertices):
        """
        Get centroid of a set of vertices.

        Args:
        vertices (np array) - cyclic voronoi vertices for a single region, (N+1) x 2

        Returns:
        centroid (np array) - 2 x 1
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

    @classmethod
    def _get_centroids(cls, voronoi):
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
            centroid = cls.get_centroid_of_region(region_vertices)
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
        self.update_bounding_points()
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
        artist = ax.plot(voronoi.filtered_points[0, :], voronoi.filtered_points[1, :], '.', color=point_color, alpha=point_alpha)
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

        xmin, ymin = voronoi.filtered_points.min(axis=1)
        xmax, ymax = voronoi.filtered_points.max(axis=1)
        ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
        return point_artists

    def plot_voronoi_regions(self, **kwargs):
        """
        Plot bounded voronoi regions.

        kwargs: plot formatting arguments
        """
        artists = self._plot_voronoi_regions(self.voronoi, **kwargs)

    @staticmethod
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
        if 'ax' in list(kwargs.keys()):
            ax = kwargs.pop('ax')
        else:
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

    def plot_translation_field(self, **kwargs):
        """
        Plot vector field of line-segments between initial and relaxed xy coordinates.

        kwargs: quiver plot formatting arguments
        """
        self.plot_vector_field(self.xycoords_initial, self.xycoords, **kwargs)

    def get_video(self, **kwargs):
        """
        Get HTML5 video of relaxation.

        kwargs: video formatting arguments
        """
        return RelaxationAnimation(self).get_video(**kwargs)
