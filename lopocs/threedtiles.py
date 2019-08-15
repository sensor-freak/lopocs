# -*- coding: utf-8 -*-
import json
import math

import numpy as np
from flask import make_response

from py3dtiles.feature_table import (
    FeatureTableHeader, FeatureTableBody, FeatureTable
)
from py3dtiles.pnts import PntsBody, PntsHeader, Pnts

from .utils import (
    read_uncompressed_patch, boundingbox_to_polygon, list_from_str, patch_numpoints, pairwise
)
from .conf import Config
from .database import Session

LOD_MIN = 0
LOD_MAX = 6
LOD_LEN = LOD_MAX + 1 - LOD_MIN


def ThreeDTilesInfo(table, column):

    session = Session(table, column)
    # bounding box
    box = session.boundingbox

    # number of points for the first patch
    npoints = session.approx_row_count * session.patch_size

    # srs
    srs = session.srs

    # build json
    return {
        "bounds": [box['xmin'], box['ymin'], box['zmin'],
                   box['xmax'], box['ymax'], box['zmax']],
        "numPoints": npoints,
        "srs": srs
    }


def ThreeDTilesRead(table, column, bounds, lod, format):

    session = Session(table, column)
    # offsets = [round(off, 2) for off in list_from_str(offsets)]
    box = list_from_str(bounds)
    # requested = [scales, offsets]
    stored_patches = session.lopocstable.filter_stored_output()
    schema = stored_patches['point_schema']
    pcid = stored_patches['pcid']
    # scales = [scale] * 3
    scales = stored_patches['scales']
    offsets = stored_patches['offsets']

    [tile, npoints] = get_points(session, box, lod, offsets, pcid, scales, schema, format)

    if Config.DEBUG:
        tile.sync()
        print("NPOINTS: ", npoints)

    # build flask response
    response = make_response(tile)
    response.headers['content-type'] = 'application/octet-stream'
    return response


def classification_to_rgb(points):
    """
    map LAS Classification to RGB colors.
    See LAS spec for codes :
    http://www.asprs.org/wp-content/uploads/2010/12/asprs_las_format_v11.pdf

    :param points: points as a structured numpy array
    :returns: numpy.record with dtype [('Red', 'u1'), ('Green', 'u1'), ('Blue', 'u1')])
    """
    # building (brown)
    building_mask = (points['Classification'] == 6).astype(np.int)
    red = building_mask * 186
    green = building_mask * 79
    blue = building_mask * 63
    # high vegetation (green)
    veget_mask = (points['Classification'] == 5).astype(np.int)
    red += veget_mask * 140
    green += veget_mask * 156
    blue += veget_mask * 8
    # medium vegetation
    veget_mask = (points['Classification'] == 4).astype(np.int)
    red += veget_mask * 171
    green += veget_mask * 200
    blue += veget_mask * 116
    # low vegetation
    veget_mask = (points['Classification'] == 3).astype(np.int)
    red += veget_mask * 192
    green += veget_mask * 213
    blue += veget_mask * 160
    # water (blue)
    water_mask = (points['Classification'] == 9).astype(np.int)
    red += water_mask * 141
    green += water_mask * 179
    blue += water_mask * 198
    # ground (light brown)
    grd_mask = (points['Classification'] == 2).astype(np.int)
    red += grd_mask * 226
    green += grd_mask * 230
    blue += grd_mask * 229
    # Unclassified (grey)
    grd_mask = (points['Classification'] == 1).astype(np.int)
    red += grd_mask * 176
    green += grd_mask * 185
    blue += grd_mask * 182

    rgb_reduced = np.c_[red, green, blue]
    rgb = np.array(np.core.records.fromarrays(rgb_reduced.T, dtype=cdt))
    return rgb


cdt = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8)])
pdt = np.dtype([('X', np.float32), ('Y', np.float32), ('Z', np.float32)])


# Get points from the database and convert them into 3DTiles file format
def get_points(session, box, lod, offsets, pcid, scales, schema, format):
    print( 'session: {0}'.format(session))
    print( 'box: {0}'.format(box))
    print( 'lod: {0}'.format(lod))
    print( 'offsets: {0}'.format(offsets))
    print( 'scales: {0}'.format(scales))
    print( 'schema: {0}'.format(schema))
    print( 'format: {0}'.format(format))

    sql = sql_query(session, box, pcid, lod)
    if Config.DEBUG:
        print(sql)

    pcpatch_wkb = session.query(sql)[0][0]
    points, npoints = read_uncompressed_patch(pcpatch_wkb, schema)
    print( 'uncompressed patch lod {1}: {0} pts'.format(npoints, lod))
    fields = points.dtype.fields.keys()
    print('Fields: {0}'.format(fields))
    for f in fields:
        print('{0} - {1}'.format(f, points[f][0]))

    if 'Red' in fields:
        if max(points['Red']) > 255:
            # normalize
            rgb_reduced = np.c_[points['Red'] % 255, points['Green'] % 255, points['Blue'] % 255]
            rgb = np.array(np.core.records.fromarrays(rgb_reduced.T, dtype=cdt))
        else:
            rgb = points[['Red', 'Green', 'Blue']].astype(cdt)
    elif 'Classification' in fields:
        rgb = classification_to_rgb(points)
    else:
        # No colors
        # FIXME: compute color gradient based on elevation
        rgb_reduced = np.zeros((npoints, 3), dtype=int)
        rgb = np.array(np.core.records.fromarrays(rgb_reduced, dtype=cdt))
    #print(rgb)

    quantized_points_r = np.c_[
        points['X'] * scales[0],
        points['Y'] * scales[1],
        points['Z'] * scales[2]
    ]
    #print('{0}'.format(quantized_points_r))

    quantized_points = np.array(np.core.records.fromarrays(quantized_points_r.T, dtype=pdt))
    #print('{0}'.format(quantized_points))
    #print('{0}'.format(rgb))

    results = ''
    if format == 'pnts':
        results = format_pnts(quantized_points, npoints, rgb, offsets)
    if format == 'pts':
        results = format_pts(quantized_points, npoints, rgb, offsets)
    return results


# Convert the points into simple PTS format
def format_pts(quantized_points, npoints, rgb, offsets):
    tile = '"X" "Y" "Z"\n'

    for ptidx in range(npoints):
        tile += '{0} {1} {2}\n' \
                .format(quantized_points[ptidx][0] + offsets[0],
                        quantized_points[ptidx][1] + offsets[1],
                        quantized_points[ptidx][2] + offsets[2],
                        rgb[ptidx][0], rgb[ptidx][1], rgb[ptidx][2])

    #tile = '{0}\n'.format(quantized_points.shape)
    #tile += '{0}\n'.format(quantized_points[0])
    return [tile, npoints]


# Convert the points into a 3DTiles structure (apparently to be formatted as pnts)
def format_pnts(quantized_points, npoints, rgb, offsets):
    fth = FeatureTableHeader.from_dtype(
        quantized_points.dtype, rgb.dtype, npoints
    )
    ftb = FeatureTableBody()
    ftb.positions_itemsize = fth.positions_dtype.itemsize
    ftb.colors_itemsize = fth.colors_dtype.itemsize
    ftb.positions_arr = quantized_points.view(np.uint8)
    ftb.colors_arr = rgb.view(np.uint8)

    ft = FeatureTable()
    ft.header = fth
    ft.body = ftb

    # tile
    tb = PntsBody()
    tb.feature_table = ft
    th = PntsHeader()
    tile = Pnts()
    tile.body = tb
    tile.header = th
    tile.body.feature_table.header.rtc = offsets

    return [tile.to_array().tostring(), npoints]


def sql_query(session, box, pcid, lod):
    poly = boundingbox_to_polygon(box)

    maxppp = session.lopocstable.max_points_per_patch
    # FIXME: need to be cached
    patch_size = session.patch_size

    if maxppp:
        range_min = 1
        range_max = maxppp
    else:
        # FIXME: may skip some points if patch_size/lod_len is decimal
        # we need to fix either here or at loading with the patch_size and lod bounds
        range_min = lod * int(patch_size / LOD_LEN) + 1
        range_max = (lod + 1) * int(patch_size / LOD_LEN)

    # build the sql query
    sql_limit = ""
    maxppq = session.lopocstable.max_patches_per_query
    if maxppq:
        sql_limit = " limit {0} ".format(maxppq)

    if Config.USE_MORTON:
        sql = ("select pc_union("
               "pc_filterbetween( "
               "pc_range({0}, {4}, {5}), 'Z', {6}, {7} )) from "
               "(select {0} from {1} "
               "where pc_intersects({0}, st_geomfromtext('polygon (("
               "{2}))',{3})) order by morton {8})_;"
               .format(session.column, session.table,
                       poly, session.srsid, range_min, range_max,
                       box[2] - 0.1, box[5] + 0.1, sql_limit,
                       pcid))
    else:
        sql = ("select pc_compress(pc_transform(pc_union("
               "pc_filterbetween( "
               "pc_range({0}, {4}, {5}), 'Z', {6}, {7} )), {9}), 'laz') from "
               "(select {0} from {1} where pc_intersects({0}, "
               "st_geomfromtext('polygon (({2}))',{3})) {8})_;"
               .format(session.column, session.table,
                       poly, session.srsid, range_min, range_max,
                       box[2], box[5], sql_limit,
                       pcid))

    return sql


def buildbox(bbox):
    width = bbox[3] - bbox[0]
    depth = bbox[4] - bbox[1]
    height = bbox[5] - bbox[2]
    midx = bbox[0] + width / 2
    midy = bbox[1] + depth / 2
    midz = bbox[2] + height / 2

    box = [midx, midy, midz]
    box.append(width / 2.0)
    box.append(0.0)
    box.append(0.0)
    box.append(0.0)
    box.append(depth / 2.0)
    box.append(0.0)
    box.append(0.0)
    box.append(0.0)
    box.append(height / 2.0)
    return box


def build_hierarchy_from_pg(session, baseurl, bbox):

    stored_patches = session.lopocstable.filter_stored_output()
    pcid = stored_patches['pcid']
    offsets = stored_patches['offsets']
    tileset = {}
    tileset["asset"] = {"version": "0.0"}
    tileset["geometricError"] = math.sqrt(
        (bbox[3] - bbox[0]) ** 2 + (bbox[4] - bbox[1]) ** 2 + (bbox[5] - bbox[2]) ** 2
    )
    if Config.DEBUG:
        print('tileset geometricErroc', tileset["geometricError"])

    bvol = {}
    bvol["box"] = buildbox(bbox)

    lod_str = "lod={0}".format(LOD_MIN)
    bounds = ("bounds=[{0},{1},{2},{3},{4},{5}]"
              .format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))
    resource = "{}.{}".format(session.table, session.column)

    base_url = "{0}/3dtiles/{1}/read.pnts".format(baseurl, resource)
    url = (
        "{0}?{1}&{2}"
        .format(base_url, lod_str, bounds)
    )

    GEOMETRIC_ERROR = tileset["geometricError"]

    root = {}
    root["refine"] = "add"
    root["boundingVolume"] = bvol
    root["geometricError"] = GEOMETRIC_ERROR / 20
    root["content"] = {"url": url}

    lod = 1
    children_list = []
    for bb in split_bbox(bbox):
        json_children = children(
            session, baseurl, offsets, bb, lod, pcid, GEOMETRIC_ERROR / 40
        )
        if len(json_children):
            children_list.append(json_children)

    if len(children_list):
        root["children"] = children_list

    tileset["root"] = root

    return json.dumps(tileset, indent=2, separators=(',', ': '))


def build_children_section(session, baseurl, offsets, bbox, err, lod):

    cjson = {}

    lod = "lod={0}".format(lod)
    bounds = ("bounds=[{0},{1},{2},{3},{4},{5}]"
              .format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))

    resource = "{}.{}".format(session.table, session.column)
    baseurl = "{0}/3dtiles/{1}/read.pnts".format(baseurl, resource)
    url = "{0}?{1}&{2}".format(baseurl, lod, bounds)

    bvol = {}
    bvol["box"] = buildbox(bbox)

    cjson["boundingVolume"] = bvol
    cjson["geometricError"] = err
    cjson["content"] = {"url": url}

    return cjson


def split_bbox(bbox):
    width = bbox[3] - bbox[0]
    length = bbox[4] - bbox[1]
    height = bbox[5] - bbox[2]
    size = max( width, length, height)
    print( "{} / {} / {} / {}".format( width, length, height, size))

    if width < size/2:
        xrange = [bbox[0], bbox[3]]
    else:
        xrange = [bbox[0], bbox[0] + width/2, bbox[3]]

    if length < size/2:
        yrange = [bbox[1], bbox[4]]
    else:
        yrange = [bbox[1], bbox[1] + length/2, bbox[4]]

    if height < size/2:
        zrange = [bbox[2], bbox[5]]
    else:
        zrange = [bbox[2], bbox[2] + height/2, bbox[5]]

    bboxes = []
    for x1, x2 in pairwise(xrange):
        for y1, y2 in pairwise(yrange):
            for z1, z2 in pairwise(zrange):
                bboxes.append( [ x1, y1, z1, x2, y2, z2 ])

    print(bboxes)
    return bboxes


def children(session, baseurl, offsets, bbox, lod, pcid, err):

    # run sql
    sql = sql_query(session, bbox, pcid, lod)
    pcpatch_wkb = session.query(sql)[0][0]

    json_me = {}
    if lod <= LOD_MAX and pcpatch_wkb:
        npoints = patch_numpoints(pcpatch_wkb)
        if npoints > 0:
            json_me = build_children_section(session, baseurl, offsets, bbox, err, lod)

        lod += 1

        children_list = []
        if lod <= LOD_MAX:
            for bb in split_bbox(bbox):
                json_children = children(
                    session, baseurl, offsets, bb, lod, pcid, err / 2
                )

                if len(json_children):
                    children_list.append(json_children)

        if len(children_list):
            json_me["children"] = children_list

    return json_me
