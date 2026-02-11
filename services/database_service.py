import psycopg2
from psycopg2 import pool
from config.settings import get_settings

settings = get_settings()

class DatabaseService:
    _connection_pool = None

    @classmethod
    def initialize_pool(cls):
        """Initialize the connection pool"""
        if cls._connection_pool is None:
            try:
                cls._connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1,
                    20,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    database=settings.DB_NAME
                )
                print("Database connection pool initialized successfully")
            except (Exception, psycopg2.DatabaseError) as error:
                print(f"Error while connecting to PostgreSQL: {error}")
                raise error

    @classmethod
    def get_connection(cls):
        """Get a connection from the pool"""
        if cls._connection_pool is None:
            cls.initialize_pool()
        return cls._connection_pool.getconn()

    @classmethod
    def release_connection(cls, connection):
        """Release a connection back to the pool"""
        if cls._connection_pool:
            cls._connection_pool.putconn(connection)

    @classmethod
    def execute_query(cls, query, params=None, fetch=False):
        """Execute a query and optionally fetch results"""
        conn = cls.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    result = cursor.fetchall()
                else:
                    conn.commit()
                    result = cursor.rowcount
            return result
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error executing query: {error}")
            conn.rollback()
            raise error
        finally:
            cls.release_connection(conn)

    @classmethod
    def close_all_connections(cls):
        """Close all connections in the pool"""
        if cls._connection_pool:
            cls._connection_pool.closeall()
            print("PostgreSQL connection pool is closed")

# Usage Example:
# result = DatabaseService.execute_query("SELECT * FROM \"exams-candidate\" WHERE id = %s", (1,), fetch=True)
