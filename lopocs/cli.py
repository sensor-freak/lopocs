#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re
import sys
import shlex
import json
from zipfile import ZipFile
from datetime import datetime
from pathlib import Path
from subprocess import check_call, call, check_output, CalledProcessError, DEVNULL
from urllib import parse
from ilock import ILock

import click
import requests
from flask_cors import CORS
from pyproj import Proj, transform

from lopocs import __version__
from lopocs import create_app, greyhound, threedtiles
from lopocs.database import Session, LopocsException
from lopocs.potreeschema import potree_schema
from lopocs.potreeschema import potree_page
from lopocs.cesium import cesium_page
from lopocs.utils import compute_scale_for_cesium, normalize_names


samples = {
    'airport': 'https://github.com/PDAL/data/raw/master/liblas/LAS12_Sample_withIntensity_Quick_Terrain_Modeler.laz',
    'sthelens': 'https://github.com/PDAL/data/raw/master/liblas/MtStHelens.laz',
    'lyon': (3946, 'http://3d.oslandia.com/lyon.laz')
}


# Used to synchronize execution of the create_pointcloud_lopocs_table function,
# to avoid concurrency problems when running multiple load commands in parallel
global_setup_lock = ILock('LOPoCS DB Lock')


PDAL_PIPELINE = """
{{
"pipeline": [
    {{
        "type": "{data_reader}",
        "filename":"{realfilename}"
        {header}
        {flt_skip}
    }},
    {{
        "type": "filters.chipper",
        "capacity": "{capacity}"
    }},
    {reproject}
    {{
        "type": "filters.mortonorder",
        "reverse": "true"
    }},
    {{
        "type":"writers.pgpointcloud",
        "connection":"dbname={pg_name} host={pg_host} port={pg_port} user={pg_user} password={pg_password}",
        "schema": "{schema}",
        "table":"{tab}",
        "compression":"none",
        "srid":"{srid}",
        "overwrite":"{overwrite}",
        "column": "{column}",
        "scale_x": "{scale[0]}",
        "scale_y": "{scale[1]}",
        "scale_z": "{scale[2]}",
        "offset_x": "{offset[0]}",
        "offset_y": "{offset[1]}",
        "offset_z": "{offset[2]}"
    }}
]
}}"""


def fatal(message):
    '''print error and exit'''
    click.echo('\nFATAL: {}'.format(message), err=True)
    sys.exit(1)


def pending(msg, nl=False):
    click.echo('[{}] {} ... '.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        msg
    ), nl=nl)


def green(message):
    click.secho(message.replace('\n', ''), fg='green')


def ok(mess=None):
    if mess:
        click.secho('{} : '.format(mess.replace('\n', '')), nl=False)
    click.secho('ok', fg='green')


def ko(mess=None):
    if mess:
        click.secho('{} : '.format(mess.replace('\n', '')), nl=False)
    click.secho('ko', fg='red')

def warn(mess=None):
    if mess:
        click.secho('{} : '.format(mess.replace('\n', '')), nl=False)
    click.secho('warning', fg='cyan')


def finished(msg):
    click.echo('[{}] {}'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        msg
    ))


def download(label, url, dest):
    '''
    download url using requests and a progressbar
    '''
    r = requests.get(url, stream=True)
    length = int(r.headers['content-length'])

    chunk_size = 512
    iter_size = 0
    with io.open(dest, 'wb') as fd:
        with click.progressbar(length=length, label=label) as bar:
            for chunk in r.iter_content(chunk_size):
                fd.write(chunk)
                iter_size += chunk_size
                bar.update(chunk_size)


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('LOPoCS version {}'.format(__version__))
    click.echo('')
    ctx.exit()


@click.group()
@click.option('--version', help='show version', is_flag=True, expose_value=False, callback=print_version)
def cli():
    '''lopocs command line tools'''
    pass


@click.option('--host', help='The hostname to listen on (default is 127.0.0.1)',
              default='127.0.0.1', type=str)
@click.option('--port', help='The port to listen on (default is 5000)',
              default=5000, type=int)
@cli.command()
def serve(host, port):
    '''run lopocs server (development usage)'''
    app = create_app()
    CORS(app)
    app.run(host=host, port=port)


def cmd_rt(message, command):
    '''wrapper around call function
    '''
    click.echo('{} ... '.format(message), nl=False)
    rt = call(command, shell=True)
    if rt != 0:
        ko()
        return
    ok()


def cmd_output(message, command):
    '''wrapper check_call function
    '''
    click.echo('{} ... '.format(message), nl=False)
    try:
        output = check_output(shlex.split(command)).decode()
        green(output)
    except Exception as exc:
        ko(str(exc))


def cmd_pg(message, request):
    '''wrapper around a session query
    '''
    click.echo('{} ... '.format(message), nl=False)
    try:
        result = Session.query(request)
        if not result:
            raise Exception('Not found')
        green(result[0][0])
    except Exception as exc:
        ko(str(exc))


@cli.command()
def check():
    '''check lopocs configuration and dependencies'''
    try:
        app = create_app()
    except Exception as exc:
        fatal(str(exc))

    if not app:
        fatal("it appears that you don't have any configuration file")

    # Lopocs version
    click.echo('LOPoCS version ... ', nl=False)
    green(__version__)
    cmd_output('LOPoCS commit', 'cat /code/lopocs-git-rev.txt')

    # pdal
    cmd_output('Pdal', 'pdal-config --version')
    cmd_rt('Pdal plugin pgpointcloud', "test -e `pdal-config --plugin-dir`/libpdal_plugin_writer_pgpointcloud.so")

    # postgresql and extensions
    cmd_pg('PostgreSQL', 'show server_version')
    cmd_pg('PostGIS extension', "select default_version from pg_available_extensions where name = 'postgis'")
    cmd_pg('PgPointcloud extension', "select default_version from pg_available_extensions where name = 'pointcloud'")
    cmd_pg('PgPointcloud-PostGIS extension', "select default_version from pg_available_extensions where name = 'pointcloud_postgis'")


@click.option('--skip-index-creation', help='If True, skip creation/update of indices. Use this when performing bulk insert of multiple pointcloud files.', type=bool, default=False, flag_value=True)
@click.option('--translate', help='Constructs a PDAL filter performing a simple xyz-transformation of the input data. (Example: "10000, 0, 0")', type=str, default=None)
@click.option('--data-reader', help='The PDAL driver to be used for reading the input file (e.g. readers.text)', type=str)
@click.option('--data-header', help='Data header containing additional arguments for the PDAL reader', type=str)
@click.option('--data-skip', help='Number of lines to ignore at the beginning of the file.', type=int, default=0)
@click.option('--data-mode', help='target column overwrite behaviour when target column exists', default='fail', type=click.Choice(['overwrite', 'append', 'fail']))
@click.option('--table', required=True, help='table name to store pointclouds, considered in public schema if no prefix provided')
@click.option('--column', help='column name to store patches`', default="points", type=str)
@click.option('--work-dir', type=click.Path(exists=True), required=True, help='working directory where temporary files will be saved')
@click.option('--capacity', type=int, default=400, help='number of points in a pcpatch')
@click.option('--potree', 'usewith', help='load data for use with greyhound/potree', flag_value='potree')
@click.option('--cesium', 'usewith', help='load data for use with 3dtiles/cesium (data will be re-projected)', flag_value='cesium')
@click.option('--native', 'usewith', help='load data for use with 3dtiles/native SRS (no re-projection)', default=True, flag_value='native')
@click.option('--srid', help='set Spatial Reference Identifier (EPSG code) for the source file', default=0, type=int)
@click.argument('filename', type=click.Path(exists=True))
@cli.command()
def load(filename, table, column, work_dir, capacity, usewith, srid, data_mode, data_header, data_skip, data_reader, translate, skip_index_creation):
    '''load pointclouds data using pdal and add metadata needed by lopocs'''
    _load(filename, table, column, work_dir, capacity, usewith, srid, data_mode, data_header, data_skip, data_reader, translate, skip_index_creation)


def _load(filename, table, column, work_dir, capacity, usewith, srid=0, data_mode='fail', data_header='', data_skip=0, data_reader='', translate=None, skip_index_creation=False):
    '''load pointclouds data using pdal and add metadata needed by lopocs'''
    # intialize flask application
    app = create_app()

    filename = Path(filename)
    work_dir = Path(work_dir)
    extension = filename.suffix[1:].lower()
    # laz uses las reader in PDAL
    extension = extension if extension != 'laz' else 'las'
    # txt uses text reader in PDAL
    extension = extension if extension != 'txt' else 'text'
    basename = filename.stem
    basedir = filename.parent
    table, column = normalize_names( table, column)

    try:
        with global_setup_lock:
            pending('Creating metadata table')
            Session.create_pointcloud_lopocs_table()
    finally:
        ok()

    # When using text files assume they have no header, so specify a header here
    # (These local variable is used to prepare PDAL_PIPELINE below!)
    header = ', "header": "{}"'.format(data_header) if (data_header is not None) & (data_header != '') else ''

    # Prepare the reader to be used
    data_reader = data_reader if (data_reader is not None) & (data_reader != '') else "readers.{}".format(extension)
    flt_skip = ', "skip": {}'.format(data_skip) if data_skip != 0 else ''

    # tablename should be always prefixed
    if '.' not in table:
        table = 'public.{}'.format(table)

    # Check if the target table already contains some values
    overwrite = 'false'
    try:
        # Assume that the estimated patch size can only be determined when the first patch contains some points
        numpoints = Session( table, column).patch_size
    except LopocsException as e:
        pass
    if 'numpoints' in locals():
        if numpoints > 0:
            if data_mode == 'fail':
                fatal('The target column exists and contains data, specify data-mode to override')
            if data_mode == 'overwrite':
                overwrite = 'true'

    pending('Reading summary with PDAL')
    json_path = os.path.join(
        str(work_dir.resolve()),
        '{basename}_{table}_pipeline.json'.format(**locals()))

    # Try to generate the summary. This might fail, epsecially for txt files
    options = ('--driver {} '.format(str(data_reader))) if data_reader else ''
    options += ('--{}.header="{}" '.format(str(data_reader), str(data_header))) if data_header else ''
    options += ('--{}.skip={}'.format(str(data_reader), str(data_skip))) if data_skip else ''
    options += ('--{}.override_srs={}'.format(str(data_reader), str(srid))) if srid == 0 else ''
    cmd = "pdal info {1} --summary {0}".format(filename, options)
    try:
        output = check_output(shlex.split(cmd))
        summary = json.loads(output.decode())['summary']
        ok()
    except CalledProcessError as e:
        warn(str(e))
        summary = []

    if 'srs' not in summary and not srid:
        fatal('Unable to find the spatial reference system, please provide a SRID with option --srid')

    if not srid:
        # find authority code in wkt string
        srid = re.findall('EPSG","(\d+)"', summary['srs']['wkt'])[-1]

    p = Proj(init='epsg:{}'.format(srid))
    if p.is_latlong():
        # geographic
        scale = (1e-6, 1e-6, 1e-2)
    else:
        # projection or geocentric
        scale = (0.01, 0.01, 0.01)

    if 'bounds' in summary:
        # A bounding box is given in the summary, so apply some scaling...
        offset = ( summary['bounds']['minx'] + (summary['bounds']['maxx'] - summary['bounds']['minx']) / 2,
                   summary['bounds']['miny'] + (summary['bounds']['maxy'] - summary['bounds']['miny']) / 2,
                   summary['bounds']['minz'] + (summary['bounds']['maxz'] - summary['bounds']['minz']) / 2)
    else:
        # The summary has no bounding box, so do not scale at all
        offset = (0, 0, 0)
        scale = (1, 1, 1)

    reproject = ""

    if usewith == 'cesium':
        from_srid = srid
        # cesium only use epsg:4978, so we must reproject before loading into pg
        srid = 4978

        reproject = """
        {{
           "type":"filters.reprojection",
           "in_srs":"EPSG:{from_srid}",
           "out_srs":"EPSG:{srid}"
        }},""".format(**locals())
        # transform bounds in new coordinate system
        pini = Proj(init='epsg:{}'.format(from_srid))
        pout = Proj(init='epsg:{}'.format(srid))
        # recompute offset in new space and start at 0
        pending('Reprojected bounds', nl=True)
        # xmin, ymin, zmin = transform(pini, pout, offset_x, offset_y, offset_z)
        xmin, ymin, zmin = transform(pini, pout, summary['bounds']['minx'], summary['bounds']['miny'], summary['bounds']['minz'])
        xmax, ymax, zmax = transform(pini, pout, summary['bounds']['maxx'], summary['bounds']['maxy'], summary['bounds']['maxz'])
        offset = (xmin, ymin, zmin)
        click.echo('{} < x < {}'.format(xmin, xmax))
        click.echo('{} < y < {}'.format(ymin, ymax))
        click.echo('{} < z < {}  '.format(zmin, zmax), nl=False)
        ok()
        pending('Computing best scales for cesium')
        # override scales for cesium if possible we try to use quantized positions
        scale = ( min(compute_scale_for_cesium(xmin, xmax), 1),
                  min(compute_scale_for_cesium(ymin, ymax), 1),
                  min(compute_scale_for_cesium(zmin, zmax), 1))
        ok('[{}, {}, {}]'.format(scale[0], scale[1], scale[2]))


    # Add a transformation filter, if requested
    if (translate is not None) and (translate != ''):
        pending('Generating translation filter')
        trans = translate.split(' ')
        if len(trans) == 3:
            reproject += """
            {{
              "type": "filters.transformation",
              "matrix": "1 0 0 {0} 0 1 0 {1} 0 0 1 {2} 0 0 0 1"
            }},
            """.format(trans[0], trans[1], trans[2])
            ok();
        else:
            fatal('Transformation arguments could not be applied. Use something like --translate "3000000 0 0".')

    pg_host = app.config['PG_HOST']
    pg_name = app.config['PG_NAME']
    pg_port = app.config['PG_PORT']
    pg_user = app.config['PG_USER']
    pg_password = app.config['PG_PASSWORD']
    realfilename = str(filename.resolve())
    schema, tab = table.split('.')

    pending('Loading point clouds into database')

    with io.open(json_path, 'w') as json_file:
        json_file.write(PDAL_PIPELINE.format(**locals()))

    cmd = "pdal pipeline {}".format(json_path)

    try:
        check_call(shlex.split(cmd), stderr=DEVNULL, stdout=DEVNULL)
    except CalledProcessError as e:
        fatal(str(e))
    ok()

    if not skip_index_creation:
        _update_table_indices( table, column, 128, srid, scale, offset)

    finished('Loading into {} completed'.format(table))


def _update_table_indices( table, column, morton_size, srid, scale, offset):
    '''
    Updated the indices for the given table and column
    '''
    schema, tab = table.split('.')
    pending("Creating indexes")
    Session.execute("""
        create index if not exists {tab}_env_idx on {table} using gist(pc_envelopegeometry({column}));
        alter table {table} add column if not exists morton bigint;
        select Morton_Update('{table}', '{column}', 'morton', {morton_size}, TRUE);
        create index if not exists {tab}_morton_idx on {table}(morton);
    """.format(**locals()))
    ok()

    pending("Adding metadata for lopocs")
    Session.update_metadata(
        table, column, srid,
        scale[0], scale[1], scale[2],
        offset[0], offset[1], offset[2]
    )
    ok()


@click.option('--table', required=True, help='table name to store pointclouds, considered in public schema if no prefix provided')
@click.option('--column', help="column name to store patches", default="points", type=str)
@click.option('--work-dir', type=click.Path(exists=True), required=True, help="working directory where temporary files will be saved")
@click.option('--server-url', type=str, help="server url for lopocs", default="http://localhost:5000")
@click.option('--potree', 'usewith', help="build tileset for use with greyhound/potree", flag_value='potree')
@click.option('--cesium', 'usewith', help="build tileset for use with 3dtiles/cesium ", flag_value='cesium')
@click.option('--native', 'usewith', help="build tileset for use with 3dtiles/native SRS ", default=True, flag_value='native')
@click.option('--lod-min', type=int, default=1, help='The LoD for the root level')
@click.option('--lod-max', type=int, default=5, help='The LoD for the most detailed level')
@cli.command()
def tileset(table, column, server_url, work_dir, usewith, lod_min, lod_max):
    """
    (Re)build a tileset.json for a given table.
    TODO: Specify the LoDs for which to generate the tileset.
    TODO: Check function of this feature when using --potree
    """
    # intialize flask application
    app = create_app()

    work_dir = Path(work_dir)

    if '.' not in table:
        table = 'public.{}'.format(table)

    lpsession = Session(table, column)

    # initialize range for level of details
    fullbbox = lpsession.boundingbox
    bbox = [
        fullbbox['xmin'], fullbbox['ymin'], fullbbox['zmin'],
        fullbbox['xmax'], fullbbox['ymax'], fullbbox['zmax']
    ]

    if usewith == 'potree':
        # add schema currently used by potree (version 1.5RC)
        Session.add_output_schema(
            table, column, scale[0], scale[1], scale[2],
            offset[0], offset[1], offset[2], srid, potree_schema
        )
        cache_file = (
            "{0}_{1}_{2}_{3}_{4}.hcy".format(
                lpsession.table,
                lpsession.column,
                lod_min,
                lod_max,
                '_'.join(str(e) for e in bbox)
            )
        )
        pending("Building greyhound hierarchy")
        new_hcy = greyhound.build_hierarchy_from_pg(
            lpsession, lod_min, lod_max, bbox
        )
        greyhound.write_in_cache(new_hcy, cache_file)
        ok()
        create_potree_page(str(work_dir.resolve()), server_url, table, column)

    if usewith == 'cesium':
        pending('Building tileset from database')
        hcy = threedtiles.build_hierarchy_from_pg(
            lpsession, server_url, bbox, lod_min, lod_max
        )
        ok()

        tileset = os.path.join(str(work_dir.resolve()), 'tileset-{}.{}.json'.format(table, column))
        pending('Writing tileset to disk')
        with io.open(tileset, 'wb') as out:
            out.write(hcy.encode())
        ok()
        create_cesium_page(str(work_dir.resolve()), table, column)

    if usewith == 'native':
        pending("Building 3Dtiles tileset (native)")
        hcy = threedtiles.build_hierarchy_from_pg(
            lpsession, server_url, bbox, lod_min, lod_max
        )
        ok()

        tileset = os.path.join(str(work_dir.resolve()), 'tileset-{}.{}.json'.format(table, column))
        pending('Writing tileset to disk')
        with io.open(tileset, 'wb') as out:
            out.write(hcy.encode())
        ok()


def create_potree_page(work_dir, server_url, tablename, column):
    '''Create an html demo page with potree viewer
    '''
    # get potree build
    potree = os.path.join(work_dir, 'potree')
    potreezip = os.path.join(work_dir, 'potree.zip')
    if not os.path.exists(potree):
        download('Getting potree code', 'http://3d.oslandia.com/potree.zip', potreezip)
        # unzipping content
        with ZipFile(potreezip) as myzip:
            myzip.extractall(path=work_dir)
    tablewschema = tablename.split('.')[-1]
    sample_page = os.path.join(work_dir, 'potree-{}.html'.format(tablewschema))
    abs_sample_page = str(Path(sample_page).absolute())
    pending('Creating a potree demo page : file://{}'.format(abs_sample_page))
    resource = '{}.{}'.format(tablename, column)
    server_url = server_url.replace('http://', '')
    with io.open(sample_page, 'wb') as html:
        html.write(potree_page.format(resource=resource, server_url=server_url).encode())
    ok()


def create_cesium_page(work_dir, tablename, column):
    '''Create an html demo page with cesium viewer
    '''
    cesium = os.path.join(work_dir, 'cesium')
    cesiumzip = os.path.join(work_dir, 'cesium.zip')
    if not os.path.exists(cesium):
        download('Getting cesium code', 'http://3d.oslandia.com/cesium.zip', cesiumzip)
        # unzipping content
        with ZipFile(cesiumzip) as myzip:
            myzip.extractall(path=work_dir)
    tablewschema = tablename.split('.')[-1]
    sample_page = os.path.join(work_dir, 'cesium-{}.html'.format(tablewschema))
    abs_sample_page = str(Path(sample_page).absolute())
    pending('Creating a cesium demo page : file://{}'.format(abs_sample_page))
    resource = '{}.{}'.format(tablename, column)
    with io.open(sample_page, 'wb') as html:
        html.write(cesium_page.format(resource=resource).encode())
    ok()


@cli.command()
@click.option('--sample', help="sample data available", default="airport", type=click.Choice(samples.keys()))
@click.option('--work-dir', type=click.Path(exists=True), required=True, help="working directory where sample files will be saved")
@click.option('--server-url', type=str, help="server url for lopocs", default="http://localhost:5000")
@click.option('--potree', 'usewith', help="load data for using with greyhound/potree", flag_value='potree')
@click.option('--cesium', 'usewith', help="load data for using with 3dtiles/cesium ", flag_value='cesium')
@click.option('--native', 'usewith', help="load data for using with 3dtiles/native SRS ", default=True, flag_value='native')
@click.option('--srid', help="set Spatial Reference Identifier (EPSG code) for the source file", default=0, type=int)
def demo(sample, work_dir, server_url, usewith, srid):
    '''
    download sample lidar data, load it into pgpointcloud
    '''
    if isinstance(samples[sample], (list, tuple)):
        # srid given
        srid = samples[sample][0]
        download_link = samples[sample][1]
    else:
        download_link = samples[sample]
    filepath = Path(download_link)
    pending('Using sample data {}: {}'.format(sample, filepath.name))
    dest = os.path.join(work_dir, filepath.name)
    ok()

    tmp = parse.urlparse(download_link)
    if tmp.scheme == 'file':
        dest = tmp.path;

    print('Download Link: {0}, dest: {1}, exists: {2}'.format( download_link, dest, os.path.exists(dest)))
    if not os.path.exists(dest):
        download('Downloading sample', download_link, dest)

    # now load data
    if srid & srid!=0:
        _load(dest, sample, 'points', work_dir, 400, usewith, srid=srid)
    else:
        _load(dest, sample, 'points', work_dir, 400, usewith)

    # build the tileset file
    tileset(sample, 'points', server_url, work_dir, usewith)

    click.echo(
        'Now you can test lopocs server by executing "lopocs serve"'
        .format(sample)
    )


@cli.command(name='update-index')
@click.option('--table', required=True, help='table name to store pointclouds, considered in public schema if no prefix provided')
@click.option('--column', help="column name to store patches. (Default='points')", default="points", type=str)
@click.option('--morton-size', help="column value for the size argument of the morton index.", default=128, type=int)
@click.option('--srid', help="set Spatial Reference Identifier (EPSG code) for the data set", default=0, type=int)
def update_index(table, column, morton_size, srid):
    '''
    Update the indices for a resource given by table and column name
    '''
    # intialize flask application
    app = create_app()

    table, column = normalize_names( table, column)

    # tablename should be always prefixed
    if '.' not in table:
        table = 'public.{}'.format(table)

    # See if the table is already registered.
    # If not, the srid must be provided by the user.
    try:
        srid = Session( table, column).srsid
    except LopocsException:
        if srid == 0:
            raise LopocsException('SRID not determined, specifiy SRID in command line (table {0}, column {1})'.format(table, column))

    scale = ( 1.0, 1.0, 1.0)
    offset = ( 0.0, 0.0, 0.0)
    _update_table_indices( table, column, morton_size, srid, scale, offset)

    finished('Updating indexes for {} completed'.format(table))
