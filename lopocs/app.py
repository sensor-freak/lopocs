# -*- coding: utf-8 -*-
from flask_restplus import Api, Resource, reqparse

from .greyhound import GreyhoundInfo, GreyhoundRead, GreyhoundHierarchy
from .threedtiles import ThreeDTilesInfo, ThreeDTilesRead, ThreeDTilesResources, ThreeDTilesGetBoundsGpx, ThreeDTilesGetBoundsGeoJson
from .database import Session

api = Api(
    version='0.2',
    title='LOPoCS API',
    description='API for accessing LOPoCS'
)


# global namespace
gns = api.namespace('infos', description='Information about LOPoCS')


@gns.route("/global")
class InfosGlobal(Resource):

    def get(self):
        return "Light OpenSource PointCloud Server by Oslandia"


@gns.route("/contact")
class InfosContact(Resource):

    def get(self):
        return "infos+li3ds@oslandia.com"


@gns.route("/online")
class InfosOnline(Resource):

    def get(self):
        return "Congratulation, LOPoCS is online!!!"


@gns.route("/sources")
class Sources(Resource):

    def get(self):
        """List available resources
        """
        Session.clear_catalog()
        Session.fill_catalog()
        resp = [
            values.asjson()
            for key, values in Session.catalog.items()
        ]
        return resp


# Greyhound namespace
ghd_ns = api.namespace('greyhound', description='Greyhound protocol')


def validate_resource(resource):
    '''Resource is a table name with schema and column name combined as
    follow : schema.table.column
    '''
    if resource.count('.') != 2:
        api.abort(404, "resource must be in the form schema.table.column")

    table = resource[:resource.rfind('.')]
    column = resource.split('.')[-1]
    return table, column


@ghd_ns.route("/<resource>/info")
class Info(Resource):

    def get(self, resource):
        table, column = validate_resource(resource)
        return GreyhoundInfo(table, column)


ghd_read = reqparse.RequestParser()
ghd_read.add_argument('depthBegin', type=int, required=False)
ghd_read.add_argument('depthEnd', type=int, required=False)
ghd_read.add_argument('depth', type=int, required=False)
ghd_read.add_argument('bounds', type=str, required=False)
ghd_read.add_argument('scale', type=float, required=False)
ghd_read.add_argument('offset', type=str, required=False)
ghd_read.add_argument('schema', type=str, required=False)
ghd_read.add_argument('compress', type=bool, required=False)


@ghd_ns.route("/<resource>/read")
class Read(Resource):

    @api.expect(ghd_read, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = ghd_read.parse_args()
        return GreyhoundRead(
            table,
            column,
            args.get('offset'),
            args.get('scale'),
            args.get('bounds'),
            args.get('depth'),
            args.get('depthBegin'),
            args.get('depthEnd'),
            args.get('schema'),
            args.get('compress'))


ghd_hierarchy = reqparse.RequestParser()
ghd_hierarchy.add_argument('depthBegin', type=int)
ghd_hierarchy.add_argument('depthEnd', type=int)
ghd_hierarchy.add_argument('bounds', type=str, required=True)
ghd_hierarchy.add_argument('scale', type=float)
ghd_hierarchy.add_argument('offset', type=str)


@ghd_ns.route("/<resource>/hierarchy")
class Hierarchy(Resource):

    @ghd_ns.expect(ghd_hierarchy, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = ghd_hierarchy.parse_args()
        return GreyhoundHierarchy(
            table, column,
            args.get('bounds'),
            args.get('depthBegin'), args.get('depthEnd'),
            args.get('scale'), args.get('offset'))


# 3Dtiles namespace
threedtiles_ns = api.namespace('3dtiles', description='3DTiles format')


@threedtiles_ns.route("/resources")
class ThreeDTilesResourcesRoute(Resource):

    def get(self):
        return ThreeDTilesResources()


@threedtiles_ns.route("/<resource>/info")
class ThreeDTilesInfoRoute(Resource):

    def get(self, resource):
        table, column = validate_resource(resource)
        return ThreeDTilesInfo(table, column)


threedtiles_read = reqparse.RequestParser()
threedtiles_read.add_argument('bounds', type=str, required=True)
threedtiles_read.add_argument('lod', type=int, required=True)


@threedtiles_ns.route("/<resource>/read.pnts")
class ThreeDTilesReadPntsRoute(Resource):

    @threedtiles_ns.expect(threedtiles_read, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = threedtiles_read.parse_args()
        return ThreeDTilesRead(
            table, column,
            args.get('bounds'),
            args.get('lod'),
            'pnts'
        )


@threedtiles_ns.route("/<resource>/read.pts")
class ThreeDTilesReadPtsRoute(Resource):

    @threedtiles_ns.expect(threedtiles_read, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = threedtiles_read.parse_args()
        return ThreeDTilesRead(
            table, column,
            args.get('bounds'),
            args.get('lod'),
            'pts'
        )


desc = 'Retrieves a GPX model of the bounds of the tiles that are stored in the database.'
paramsdesc = {'resource': 'The name of the resource to be queried'}
@threedtiles_ns.route("/<resource>/get.bounds.gpx")
@threedtiles_ns.doc(description=desc, params=paramsdesc)
class ThreeDTilesGetBoundsGpxRoute(Resource):

    threedtiles_getbounds = reqparse.RequestParser()
    threedtiles_getbounds.add_argument('limit', type=int, required=False, default=4096,
                                       help='The maximum number of tiles to process. Use 0 to load all tiles (this may take some time!)')

    @threedtiles_ns.expect(threedtiles_getbounds, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = self.threedtiles_getbounds.parse_args()
        return ThreeDTilesGetBoundsGpx(
            table, column,
            args.get('limit')
        )


desc = 'Retrieves a GeoJSON model of the bounds of the tiles that are stored in the database.'
paramsdesc = {'resource': 'The name of the resource to be queried'}
@threedtiles_ns.route("/<resource>/get.bounds.geojson")
@threedtiles_ns.doc(description=desc, params=paramsdesc)
class ThreeDTilesGetBoundsGeoJsonRoute(Resource):

    threedtiles_getbounds = reqparse.RequestParser()
    threedtiles_getbounds.add_argument('limit', type=int, required=False, default=4096,
                                       help='The maximum number of tiles to process. Use 0 to load all tiles (this may take some time!)')
    threedtiles_getbounds.add_argument('bounds', type=str, required=False, default='',
                                        help='The bounds of the area to generate the overview for. If not specified, all data is used.')
    threedtiles_getbounds.add_argument('style', type=str, required=False, default='points',
                                        choices=('points', 'polygons'),
                                        help='Specify the geometry type to be generated for each tile.')

    @threedtiles_ns.expect(threedtiles_getbounds, validate=True)
    def get(self, resource):
        table, column = validate_resource(resource)
        args = self.threedtiles_getbounds.parse_args()
        return ThreeDTilesGetBoundsGeoJson(
            table, column,
            args.get('limit'),
            args.get('bounds'),
            args.get('style')
        )

