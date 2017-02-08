__author__ = 'Sebi'


#from lloyd_relaxation import LloydRelaxation
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np


class RelaxationAnimation:
    def __init__(self, lloyd_relaxation):
        self.relaxation = lloyd_relaxation
        self.bounding_box = lloyd_relaxation.bounding_box
        self.history = lloyd_relaxation.history
        self.point_lines = None
        self.centroid_lines = None
        self.ridge_lines = None
        self.region_shading = None

    def initialize_artists(self, ax,
                           point_color='black', point_size=5, point_alpha=0.5,
                           centroid_color='red', centroid_size=5, centroid_alpha=0.5,
                           line_color='black', linewidth=3, line_alpha=0.5,
                           fill_color='black', fill_alpha=0.25):
        """ Instantiate plot elements. """
        self.point_lines = ax.plot([], [], '.', color=point_color, markersize=point_size, alpha=point_alpha)
        self.centroid_lines = ax.plot([], [], '+', color=centroid_color, markersize=centroid_size, alpha=centroid_alpha)
        self.ridge_lines = [ax.plot([], [], '-', color=line_color, linewidth=linewidth, alpha=line_alpha) for _ in range(len(self.relaxation.voronoi.filtered_regions))]
        self.region_shading = ax.add_collection(PatchCollection(np.array([Polygon(np.zeros((1, 2)), closed=True)]), color=fill_color, alpha=fill_alpha))

    def draw(self, voronoi, include_points=False, include_centroids=False, include_ridges=False, include_fill=False):
        """ Update plot elements. """

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
            self.region_shading.set_paths(patches)

        return self.point_lines

    def animate(self, frame_duration=100, include_points=False, include_centroids=False,
                include_ridges=True, include_fill=True, **kwargs):
        """ Generate animation. """

        # create figure
        fig = plt.figure()
        xlim = self.bounding_box[0:2]
        ylim = self.bounding_box[2:4]
        ax = plt.axes(xlim=xlim, ylim=ylim)
        ax.set_xticks([]), ax.set_yticks([])

        # initialize plot elements
        self.initialize_artists(ax, **kwargs)

        # generate animation
        anim = animation.FuncAnimation(fig, func=self.draw, frames=self.history, interval=frame_duration, blit=True,
                                       fargs=(include_points, include_centroids, include_ridges, include_fill))

        return anim

    def get_video(self, **kwargs):
        """ Return HTML5 video of animation. """
        return self.animate(**kwargs).to_html5_video()