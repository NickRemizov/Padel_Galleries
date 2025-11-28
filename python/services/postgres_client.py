"""
Backward compatibility layer for postgres_client.
All functionality has been refactored into modular structure in services/database/
This file simply re-exports the unified PostgresClient for existing imports.
"""

from services.database import PostgresClient, db_client

__all__ = ['PostgresClient', 'db_client']
