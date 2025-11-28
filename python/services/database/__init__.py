"""
Database services package - modular PostgresClient implementation.

This package splits the monolithic PostgresClient into logical domain modules
while maintaining backward compatibility through a unified PostgresClient class.
"""

from .base_client import BaseClient
from .faces_client import FacesClient
from .people_client import PeopleClient
from .galleries_client import GalleriesClient
from .metadata_client import MetadataClient
from .recognition_client import RecognitionClient


class PostgresClient(
    BaseClient,
    FacesClient,
    PeopleClient,
    GalleriesClient,
    MetadataClient,
    RecognitionClient
):
    """
    Unified PostgresClient that combines all domain-specific clients.
    
    This maintains backward compatibility with existing code that imports:
    from services.postgres_client import db_client
    """
    pass


db_client = PostgresClient()

__all__ = [
    'PostgresClient',
    'db_client',
    'BaseClient',
    'FacesClient',
    'PeopleClient',
    'GalleriesClient',
    'MetadataClient',
    'RecognitionClient',
]
