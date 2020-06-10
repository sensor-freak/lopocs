# -*- coding: utf-8 -*-
import json
import geojson
import math
import multiprocessing as mp
from functools import partial

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
from .t2pconverter import T2PConverter


def ThreeDTilesResources():

    """List available resources
    """
    Session.clear_catalog()
    Session.fill_catalog()
    resp = [
        key[0] + '.' + key[1]
        for key, values in Session.catalog.items()
    ]
    return resp


def ThreeDTilesInfo(table, column):

    session = Session(table, column)
    # bounding box
    box = session.boundingbox

    # number of points for the first patch
    npoints = session.approx_row_count * session.patch_size

    # srs
    srs = session.srs

    sql = 'select count(*) from {}'.format(table)
    npatches = session.query(sql)[0][0]

    # build json
    return {
        "bounds": [box['xmin'], box['ymin'], box['zmin'],
                   box['xmax'], box['ymax'], box['zmax']],
        "numPoints": npoints,
        "numPatches": npatches,
        "maxPatchSize": session.lopocstable.max_points_per_patch,
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

    # When Z is empty, set some valid Z range
    if box[2] >= box[5]:
        box[5] = box[2] + 1000

    [tile, npoints] = get_points(session, box, lod, offsets, pcid, scales, schema, format)

    if Config.DEBUG:
        #tile.sync()
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


# Get points from a single patch (returned from database) and convert them into 3DTiles format
def get_points_from_patch(patch, schema, lod, scales, offsets, resultformat):
    tile = ''
    cnt = 0
    pcpatch_wkb = patch[0]
    if pcpatch_wkb:
        points, npoints = read_uncompressed_patch(pcpatch_wkb, schema)
        # print( 'uncompressed patch lod {1}: {0} pts'.format(npoints, lod))
        fields = points.dtype.fields.keys()
        # print('Fields: {0}'.format(fields))
        # for f in fields:
        #     print('{0} - {1}'.format(f, points[f][0]))

        if ('Red' in fields) & ('Green' in fields) & ('Blue' in fields):
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
            rgb_reduced = np.zeros((3, npoints), dtype=int)
            # rgb = np.array(np.core.records.fromarrays(rgb_reduced, dtype=cdt))
            # hint: Set rgb to None to avoid dumping of RGB values
            rgb = None

        quantized_points_r = np.c_[
            points['X'] * scales[0],
            points['Y'] * scales[1],
            points['Z'] * scales[2]
        ]
        #print('{0}'.format(quantized_points_r))

        quantized_points = np.array(np.core.records.fromarrays(quantized_points_r.T, dtype=pdt))

        if resultformat == 'pnts':
            tile, cnt = format_pnts(quantized_points, npoints, rgb, offsets)
        if resultformat == 'pts':
            tile, cnt = format_pts(quantized_points, npoints, rgb, offsets, points, fields)

    return tile, cnt


# Get points from the database and convert them into 3DTiles file format
def get_points(session, box, lod, offsets, pcid, scales, schema, resultformat):

    sql = sql_query(session, box, pcid, lod)
    pcpatch_result = session.query(sql)
    # print('-----------------')
    # print('result type: {}, len {}'.format(type(pcpatch_result), len(pcpatch_result)))
    # if len(pcpatch_result):
    #     print('result[0] type: {}, len {}'.format(type(pcpatch_result[0]), len(pcpatch_result[0])))
    # print('-----------------')

    with mp.Pool(processes=mp.cpu_count()) as pool:
        try:
            wrapper = partial(get_points_from_patch, schema=schema, lod=lod, scales=scales, offsets=offsets, resultformat=resultformat)
            work = pool.map_async(wrapper, pcpatch_result)
            results = work.get()
        finally:
            pool.close()
            pool.join()


    result_pts = ''
    result_cnt = 0
    # for patch in pcpatch_result:
    for tile, cnt in results:
        # tile, cnt = get_points_from_patch(patch, schema, lod, scales, offsets, resultformat)
        result_pts += tile
        result_cnt += cnt

    return result_pts, result_cnt


# Convert the points into simple PTS format
def format_pts(quantized_points, npoints, rgb, offsets, points, fields):
    header1 = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue'] if rgb is not None else ['X', 'Y', 'Z']
    header2 = [fname for fname in fields if (fname not in header1)]
    tile = '"' + '" "'.join(header1 + header2) + '"\n'

    for ptidx in range(npoints):
        if rgb is None:
            tile += '{0} {1} {2}' \
                    .format(quantized_points[ptidx][0] + offsets[0],
                            quantized_points[ptidx][1] + offsets[1],
                            quantized_points[ptidx][2] + offsets[2])
        else:
            tile += '{0} {1} {2} {3} {4} {5}' \
                    .format(quantized_points[ptidx][0] + offsets[0],
                            quantized_points[ptidx][1] + offsets[1],
                            quantized_points[ptidx][2] + offsets[2],
                            rgb[ptidx][0], rgb[ptidx][1], rgb[ptidx][2])
        for f in header2:
            tile += ' {}'.format( points[f][ptidx])
        tile += '\n'

    return [tile, npoints]


# Convert the points into a 3DTiles structure (apparently to be formatted as pnts)
def format_pnts(quantized_points, npoints, rgb, offsets):
    if rgb is not None:
        fth = FeatureTableHeader.from_dtype(quantized_points.dtype, rgb.dtype, npoints)
    else:
        fth = FeatureTableHeader.from_dtype(quantized_points.dtype, None, npoints)
    ftb = FeatureTableBody()
    ftb.positions_itemsize = fth.positions_dtype.itemsize
    ftb.positions_arr = quantized_points.view(np.uint8)
    if rgb is not None:
        ftb.colors_itemsize = fth.colors_dtype.itemsize
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
    maxppp = maxppp if maxppp else 1024

    # FIXME: need to be cached
    patch_size = session.patch_size
    LOD_MAX_ABS = 20 # The highest possible LoD value accepted
    lodlocal = max( 0, min( LOD_MAX_ABS, lod)) # The effective LoD to use
    lodnp = 2 ** (LOD_MAX_ABS - lodlocal)

    if maxppp & False:
        range_min = 1
        range_max = maxppp
    else:
        if lodnp > 0:
            range_min = 1
            range_max = max( int(maxppp / lodnp), 1)
        else:
            range_min = 1
            range_max = int(2**(lodlocal/2))

    if Config.DEBUG:
        print( 'Range: {} .. {}, maxppp: {}, lodnp: {}'.format( range_min, range_max, maxppp, lodnp))

    # build the sql query
#    sql_limit = ""
#    maxppq = session.lopocstable.max_patches_per_query
#    if maxppq:
#        sql_limit = " limit {0} ".format(maxppq)
    sql_limit = " limit {} ".format(min(2**(4+lodlocal), 1024))

    if Config.USE_MORTON:
        sql = ("""
                select 
                    pc_filterbetween( 
                        pc_intersection(
                            pc_range({0}, {4}, {5}),
                            st_geomfromtext('polygon (({2}))',{3})),
                        'Z', {6}, {7} )
                from (
                    select {0} 
                    from {1} 
                    where {0}::geometry && st_geomfromtext('polygon (({2}))',{3}) 
                    order by morton {8}
                    )_;
                """.format(session.column, session.table,
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

    if Config.DEBUG:
        print('LoD {0}, patch_size {1}, s/2^l {2}'.format(lod, patch_size, patch_size / (2 ** lod)))
        #print('Resulting SQL: ' + sql)

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


def build_hierarchy_from_pg(session, baseurl, bbox, lodmin, lodmax):

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

    lod_str = "lod={0}".format(lodmin)
    bounds = ("bounds=[{0},{1},{2},{3},{4},{5}]"
              .format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))
    resource = "{}.{}".format(session.table, session.column)

    # The URL uses the projected coordinates
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

    lodmin = lodmin + 1
    children_list = []
    for bb in split_bbox(bbox):
        json_children = children(
            session, baseurl, offsets, bb, lodmin, lodmax,
            pcid, GEOMETRIC_ERROR / 40
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

    return bboxes


def children(session, baseurl, offsets, bbox, lod, lodmax, pcid, err):

    print("children(lod {}, {})".format(lod, bbox))

    # run sql
    sql = sql_query(session, bbox, pcid, lod)
    pcpatch_wkb = session.query(sql)[0][0]

    json_me = {}
    if lod <= lodmax and pcpatch_wkb:
        npoints = patch_numpoints(pcpatch_wkb)
        if npoints > 0:
            json_me = build_children_section(session, baseurl, offsets, bbox, err, lod)

        lod += 1

        children_list = []
        if lod <= lodmax:
            for bb in split_bbox(bbox):
                json_children = children(
                    session, baseurl, offsets, bb, lod, lodmax, pcid, err / 2
                )

                if len(json_children):
                    children_list.append(json_children)

        if len(children_list):
            json_me["children"] = children_list

    return json_me


def ThreeDTilesLoad(filename, table, column, work_dir, capacity, usewith, srid=0, data_mode='fail', data_header='', data_reader=''):
    return {
        "filename": filename,
        "table": table,
        "column": column,
        "work_dir": work_dir,
        "capacity": capacity,
        "usewith": usewith,
        "srid": srid,
        "data_mode": data_mode,
        "data_header": data_header,
        "data_reader": data_reader,
        "result": False
    }


def ThreeDTilesGetBoundsGpx(table, column, limit):
    conv = T2PConverter()

    session = Session(table, column)
    bbox = session.boundingbox
    bbox['srs'] = session.srsid

    # Add the total bounding box as a track
    sql = """
        select
            st_asgeojson(st_transform(st_PointFromText('POINT({xmin} {ymin})', {srs}), 4326)),
            st_asgeojson(st_transform(st_PointFromText('POINT({xmax} {ymax})', {srs}), 4326))
          """.format(**bbox)
    bboxwgs = session.query(sql)
    ptmin = json.loads(bboxwgs[0][0])['coordinates']
    ptmax = json.loads(bboxwgs[0][1])['coordinates']
    # Why are indices 1-0 here? 0 should be X, y should be y!
    conv.add_track( ptmin[1], ptmin[0], ptmax[1], ptmax[0], bbox['zmax'])

    limitclause = 'order by morton limit {}'.format(limit) if limit != 0 else ''

    # Add tile bounds as routes
    sql = 'select st_asgeojson(st_transform({column}::geometry, 4326)) from {table} {limitclause}'.format(**locals())
    tiles = session.query(sql)
    for tile in tiles:
        tileobj = json.loads(tile[0])
        conv.add_route(tileobj['coordinates'][0][0][1], tileobj['coordinates'][0][0][0],
                       tileobj['coordinates'][0][2][1], tileobj['coordinates'][0][2][0],
                       bbox['zmax'] + 1000)

    result = conv.to_xml()

    # Make an XML response and return it
    response = make_response(result)
    response.headers['Content-Type'] = 'application/gpx+xml'
    return response


def ThreeDTilesGetBoundsGeoJson(table, column, limit, bounds, style):
    session = Session(table, column)
    bbox = session.boundingbox
    bbox['srs'] = session.srsid

    # Add the total bounding box as a Feature with geometry
    sql = """
        select
            st_asgeojson(st_transform(st_PointFromText('POINT({xmin} {ymin})', {srs}), 4326)),
            st_asgeojson(st_transform(st_PointFromText('POINT({xmax} {ymax})', {srs}), 4326))
          """.format(**bbox)
    bboxwgs = session.query(sql)
    ptmin = json.loads(bboxwgs[0][0])['coordinates']
    ptmax = json.loads(bboxwgs[0][1])['coordinates']
    geom = geojson.Polygon([[(ptmin[0], ptmin[1]),
                             (ptmax[0], ptmin[1]),
                             (ptmax[0], ptmax[1]),
                             (ptmin[0], ptmax[1]),
                             (ptmin[0], ptmin[1])]])
    outerbox = geojson.Feature(geometry=geom, bbox=[ptmin[0], ptmin[1], ptmax[0], ptmax[1]])
    outerbox.properties['description'] = 'total'

    featcoll = geojson.FeatureCollection([])
    featcoll.bbox = [ptmin[0], ptmin[1], ptmax[0], ptmax[1]]
    featcoll.features.append(outerbox)

    limitclause = 'limit {}'.format(limit) if limit != 0 else ''
    whereclause = ''
    if (bounds != ''):
        box = list_from_str(bounds)
        poly = boundingbox_to_polygon(box)
        srid = session.srsid
        whereclause = "where {column}::geometry && st_geomfromtext('polygon (({poly}))',{srid})".format(**locals())
        if style == 'polygons':
            sql = 'select st_asgeojson(st_transform(st_envelope({column}::geometry), 4326)) AS {column} FROM {table} {whereclause} ORDER BY morton {limitclause}'.format(**locals())
        else:
            sql = 'select st_asgeojson(st_transform((st_dumppoints({column}::geometry)).geom, 4326)) AS {column} FROM {table} {whereclause} ORDER BY morton {limitclause}'.format(**locals())
    else:
        if style == 'polygons':
            sql = 'select st_asgeojson(st_transform(st_envelope({column}::geometry), 4326)) AS {column} FROM {table} {whereclause} ORDER BY morton {limitclause}'.format(**locals())
        else:
            sql = 'select st_asgeojson(st_transform((select (st_dumppoints(st_union({column}::geometry))) limit 1).geom, 4326)) AS {column} FROM {table} {whereclause} GROUP BY morton {limitclause}'.format(**locals())

    tiles = session.query(sql)
    for tile in tiles:
        tileobj = json.loads(tile[0])
        tilebox = geojson.Feature(geometry=tileobj)
        tilebox.properties['description'] = 'tile'
        featcoll.features.append(tilebox)

    result = featcoll

    # Make a geojson response and return it
    response = make_response(result)
    response.headers['Content-Type'] = 'application/geojson'
    return response
