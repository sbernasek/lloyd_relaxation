

from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


class RelaxationAnimation:
    """
    Enables frame-by-frame animation of a Lloyd relaxation.

    Attributes:
    relaxation (LloydRelaxation)
    history (array like) - filtered voronoi objects constructed at each iteration of relaxation
    shader (function) - computes patch color array a list of voronoi regions. if None, use single fill_color

    Live plot elements:
    point_lines (Line2D) - xy data
    centroid_lines (Line2D) - region centroids
    ridge_lines (list of Line2D) - ridge boundaries
    region_shading (list of Polygons) - region fill patches
    """

    def __init__(self, lloyd_relaxation):
        """ Load LloydRelaxation. """
        self.relaxation = lloyd_relaxation
        self.history = lloyd_relaxation.history

        # initialize patch shader
        self.shader = None

        # initialize plot elements
        self.point_lines = None
        self.centroid_lines = None
        self.ridge_lines = None
        self.region_shading = None

    def set_shader(self, shader):
        """ Set patch shader. """
        self.shader = shader

    def get_clims(self):
        """ Return minimum and maximum values for color scale throughout animation. """
        shades = np.concatenate([self.shader(voronoi) for voronoi in self.history])
        vmin, vmax = shades.min(), shades.max()
        return vmin, vmax

    def initialize_elements(self, ax,
                           point_color='black', point_size=5, point_alpha=0.5,
                           centroid_color='red', centroid_size=5, centroid_alpha=0.5,
                           line_color='black', linewidth=3, line_alpha=0.5,
                           fill_color='black', fill_alpha=0.25,
                           boundary_color='black', boundary_width=3, boundary_alpha=0.5,
                           cmap=plt.cm.Blues, clim=None):
        """
        Initialize plot elements.

        Args:
        ax (axes object) - axis on which animation will occur
        point_color, centroid_color, line_color, fill_color, boundary_color (str) - element color options
        point_size, centroid_size, linewidth, boundary_width (float) - element sizing options
        point_alpha, centroid_alpha, line_alpha, fill_alpha, boundary_alpha (float) - element opacity options
        cmap (matplotlib colormap) - colormap used to color patches. if None, use fill_color
        clim (tuple) - bounds for color map
        """

        # add points, centroids, and ridge line elements to axes
        self.point_lines = ax.plot([], [], '.', color=point_color, markersize=point_size, alpha=point_alpha)
        self.centroid_lines = ax.plot([], [], '+', color=centroid_color, markersize=centroid_size, alpha=centroid_alpha)
        self.ridge_lines = [ax.plot([], [], '-', color=line_color, linewidth=linewidth, alpha=line_alpha) for _ in range(len(self.relaxation.voronoi.filtered_regions))]

        # instantiate patches
        patches = PatchCollection(np.array([Polygon(np.zeros((1, 2)), closed=True)]), alpha=fill_alpha)

        # set patch coloring method
        if self.shader is not None:
            patches.set_cmap(cmap)
            if clim is None:
                clim = self.get_clims()
            patches.set_clim(*clim)
            patches.set_array(np.empty(1))
        else:
            patches.set_color(fill_color)

        # add patch elements to axes
        self.region_shading = ax.add_collection(patches)

        # add border
        boundary_artist = self.add_boundary(ax, color=boundary_color, width=boundary_width, alpha=boundary_alpha)

    def add_boundary(self, ax, color='black', width=1, alpha=0.5):
        """
        Add boundary to axes.

        Args:
        ax (matplotlib axes)
        color, width, alpha - line formatting parameters

        Returns:
        artist (matplotlib artist)
        """
        boundary = self.relaxation.boundary
        boundary = np.append(boundary, boundary[:, 0].reshape(2, 1), axis=1)
        artist = ax.plot(boundary[0, :], boundary[1, :], '-', color=color, linewidth=width, alpha=alpha)
        return artist

    def update_ridge_elements(self, voronoi):
        """ Update ridge line plot elements. """
        ridges = [voronoi.vertices[region + [region[0]], :] for region in voronoi.filtered_regions]
        _ = [line[0].set_data(*ridge.T) for ridge, line in zip(ridges, self.ridge_lines)]

    def update_xy_elements(self, voronoi):
        """ Update xy data plot elements. """
        self.point_lines[0].set_data(*voronoi.filtered_points.T)

    def update_centroid_elements(self, voronoi):
        """ Update centroid plot elements. """
        centroids = self.relaxation._get_centroids(voronoi)
        self.centroid_lines[0].set_data(*centroids.T)

    def update_fill_elements(self, voronoi):
        """ Update voronoi region fill plot elements. """
        patches = [Polygon([voronoi.vertices[i] for i in region], True) for region in voronoi.filtered_regions]
        self.region_shading.set_paths(patches)
        if self.shader is not None:
            colors = self.shader(voronoi)
            self.region_shading.set_array(colors)

    def update_elements(self, voronoi,
                        include_points=False, include_centroids=False, include_ridges=False, include_fill=False):
        """
        Update plot elements for a single frame (voronoi instance).

        Args:
        voronoi (scipy.spatial.Voronoi object) - filtered regions
        include_points, include_centroids, include_ridges, include_fill (bool) - flags for plot element inclusion
        """

        if include_ridges:
            self.update_ridge_elements(voronoi)

        if include_points:
            self.update_xy_elements(voronoi)

        if include_centroids:
            self.update_centroid_elements(voronoi)

        if include_fill:
            self.update_fill_elements(voronoi)

    def animate(self, framerate=10,
                include_points=False, include_centroids=False, include_ridges=True, include_fill=True,
                shader='area', **kwargs):
        """
        Generate animation by sequentially updating plot elements with voronoi region history.

        Args:
        framerate (float) - defines animation speed in Hz
        include_points, include_centroids, include_ridges, include_fill (bool) - flags for plot element inclusion
        cmap (matplotlib colormap) - colormap used to color patches
        shader (function) - computes patch color array a list of voronoi regions. if None, use single fill_color

        kwargs: element formatting arguments

        Returns:
        anim (matplotlib.FuncAnimation)
        """

        # create and format figure
        fig = plt.figure()
        xlim, ylim = self.relaxation.get_boundary_limits()
        ax = plt.axes(xlim=xlim, ylim=ylim)
        ax.set_xticks([]), ax.set_yticks([])

        # set shading mechanism
        if shader is not None:

            # check if using a default shading mechanism
            if shader == 'area':
                shader = area_shader

            if shader == 'log_area':
                shader = partial(area_shader, log=True)

            # set shader
            self.set_shader(shader)

        # initialize plot elements
        self.initialize_elements(ax, **kwargs)

        # generate animation
        anim = animation.FuncAnimation(fig, func=self.update_elements, frames=self.history, interval=1e3/framerate,
                                       blit=False,
                                       fargs=(include_points, include_centroids, include_ridges, include_fill))

        return anim

    def get_video(self, **kwargs):
        """
        Return HTML5 video of Lloyd relaxation animation.

        kwargs: element formatting arguments

        Returns:
        html5_video
        """
        return self.animate(**kwargs).to_html5_video()


def get_polygon_area(x, y):
    """ Compute area enclosed by a set of points. """
    return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y, np.roll(x,1)))


def area_shader(voronoi, log=False):
    """
    Computes area of each voronoi region.

    Args:
    voronoi (scipy.spatial.Voronoi object)
    log (bool) - if True, use log of area

    Returns:
    areas (array like) - area of each region
    """

    # get vertices for each region
    vertices_per_region = [voronoi.vertices[region, :] for region in voronoi.filtered_regions]

    # compute areas
    areas = np.array([get_polygon_area(*vertices.T) for vertices in vertices_per_region])

    if log:
        areas = np.log10(areas)

    return areas