# -*- coding: utf-8 -*-
import gpxpy
import gpxpy.gpx


class T2PConverter(gpxpy.gpx.GPX):
    def __init__(self):
        gpxpy.gpx.GPX.__init__(self)

    def add_route(self, xmin, ymin, xmax, ymax, elevation, name=None, comment=None):
        route = gpxpy.gpx.GPXRoute()

        route.points.append(gpxpy.gpx.GPXRoutePoint(xmin, ymin, elevation))
        route.points.append(gpxpy.gpx.GPXRoutePoint(xmin, ymax, elevation))
        route.points.append(gpxpy.gpx.GPXRoutePoint(xmax, ymax, elevation))
        route.points.append(gpxpy.gpx.GPXRoutePoint(xmax, ymin, elevation))
        route.points.append(gpxpy.gpx.GPXRoutePoint(xmin, ymin, elevation))

        if name:
            route.name = name
        if comment:
            route.comment = comment

        self.routes.append(route)

    def add_track(self, xmin, ymin, xmax, ymax, elevation, name=None, comment=None):
        track = gpxpy.gpx.GPXTrack()
        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)

        segment.points.append(gpxpy.gpx.GPXTrackPoint(xmin, ymin, elevation))
        segment.points.append(gpxpy.gpx.GPXTrackPoint(xmin, ymax, elevation))
        segment.points.append(gpxpy.gpx.GPXTrackPoint(xmax, ymax, elevation))
        segment.points.append(gpxpy.gpx.GPXTrackPoint(xmax, ymin, elevation))
        segment.points.append(gpxpy.gpx.GPXTrackPoint(xmin, ymin, elevation))

        if name:
            track.name = name
        if comment:
            track.comment = comment

        self.tracks.append(track)
