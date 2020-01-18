"""
WSGI config for invest project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from invest.settings import DEBUG
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'invest.settings')

application = get_wsgi_application()

if not DEBUG:    # Running on Heroku
    from dj_static import Cling
    application = Cling(get_wsgi_application())