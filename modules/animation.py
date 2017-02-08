

#from lloyd_relaxation import LloydRelaxation
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np


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

    def initialize_elements(self, ax,
                           point_color='black', point_size=5, point_alpha=0.5,
                           centroid_color='red', centroid_size=5, centroid_alpha=0.5,
                           line_color='black', linewidth=3, line_alpha=0.5,
                           fill_color='black', fill_alpha=0.25,
                           cmap=plt.cm.Blues, vmin=0, vmax=1):
        """
        Initialize plot elements.

        Args:
        ax (axes object) - axis on which animation will occur
        point_color, centroid_color, line_color, fill_color (str) - element color options
        point_size, centroid_size, linewidth (float) - element sizing options
        point_alpha, centroid_alpha, line_alpha, fill_alpha (float) - element opacity options
        cmap (matplotlib colormap) - colormap used to color patches. if None, use fill_color
        vmin, vmax (float) - bounds for color map
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
            patches.set_clim(vmin, vmax)
            patches.set_array(np.empty(1))
        else:
            patches.set_color(fill_color)

        # add patch elements to axes
        self.region_shading = ax.add_collection(patches)

    def update_elements(self, voronoi,
                        include_points=False, include_centroids=False, include_ridges=False, include_fill=False):
        """
        Update plot elements for a single frame (voronoi instance).

        Args:
        voronoi (scipy.spatial.Voronoi object) - filtered regions
        include_points, include_centroids, include_ridges, include_fill (bool) - flags for plot element inclusion
        """

        # plot ridges
        if include_ridges:
            ridges = [voronoi.vertices[region + [region[0]], :] for region in voronoi.filtered_regions]
            _ = [line[0].set_data(*ridge.T) for ridge, line in zip(ridges, self.ridge_lines)]

        # plot points
        if include_points:
            self.point_lines[0].set_data(*voronoi.filtered_points.T)

        # plot centroids
        if include_centroids:
            centroids = LloydRelaxation._get_centroids(voronoi)
            self.centroid_lines[0].set_data(*centroids.T)

        # shade regions
        if include_fill:
            patches = [Polygon(np.array([voronoi.vertices[i] for i in region]), True) for region in voronoi.filtered_regions]
            if self.shader is not None:
                colors = self.get_patch_colors(voronoi)
                patches.set_array(colors)
            self.region_shading.set_paths(patches)

    def animate(self, framerate=10,
                include_points=False, include_centroids=False, include_ridges=True, include_fill=True,
                shader=None, **kwargs):
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

        # create figure
        fig = plt.figure()
        xlim = self.relaxation.bounding_box[0:2]
        ylim = self.relaxation.bounding_box[2:4]
        ax = plt.axes(xlim=xlim, ylim=ylim)
        ax.set_xticks([]), ax.set_yticks([])

        # initialize plot elements
        if shader is not None:
            self.set_shader(shader)
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

