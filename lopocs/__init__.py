# -*- coding: utf-8 -*-
import io
import os
import sys
from pathlib import Path

from flask import Flask, Blueprint #, DispatcherMiddleware
from yaml import load as yload
from werkzeug.middleware.proxy_fix import ProxyFix

from lopocs.app import api
from lopocs.database import Session
from lopocs.stats import Stats
from lopocs.conf import Config

# lopocs version
__version__ = '1.19'


def load_yaml_config(filename):
    """
    Open Yaml file, load content for flask config and returns it as a python dict
    """
    content = io.open(filename, 'r').read()
    return yload(content).get('flask', {})


def create_app(env='Defaults'):
    """
    Creates application.
    :returns: flask application instance
    """
    app = Flask(__name__)
    cfgfile = os.environ.get('LOPOCS_SETTINGS')
    if cfgfile:
        app.config.update(load_yaml_config(cfgfile))
    else:
        try:
            cfgfile = (Path(__file__).parent / '..' / 'conf' / 'lopocs.yml').resolve()
        except FileNotFoundError:
            app.logger.critical('no config file found !!')
            sys.exit(1)
    app.config.update(load_yaml_config(str(cfgfile)))

    app.logger.debug('loading config from {}'.format(cfgfile))

    # load extensions
    if 'URL_PREFIX' in app.config:
        blueprint = Blueprint('api', __name__, url_prefix=app.config['URL_PREFIX'])
    else:
        blueprint = Blueprint('api', __name__)

    # Apply the ProxyFix to allow uage of the application behind a reverse proxy
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    #app.wsgi_app = DispatcherMiddleware(app, {'/lopocs': app.wsgi_app})

    api.init_app(blueprint)
    app.register_blueprint(blueprint)
    Session.init_app(app)
    Config.init(app.config)

    if Config.STATS:
        Stats.init()

    return app
