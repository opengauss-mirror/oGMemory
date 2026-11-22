"""Credential-free identity of actual SQL resources used for recovery binding."""


def sql_resource_identity(conn, tables):
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(),current_schema(),inet_server_addr()::text,"
                    "inet_server_port(),(SELECT oid FROM pg_database WHERE datname=current_database())")
        database, schema, address, port, database_oid = cur.fetchone()
        resources = []
        for table in sorted(tables):
            cur.execute("SELECT n.nspname,c.relname,c.oid FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=%s::regclass", (table,))
            row = cur.fetchone()
            if row is None:
                raise ValueError('recovery table is missing')
            namespace, name, oid = row
            cur.execute("SELECT attname,format_type(atttypid,atttypmod) FROM pg_attribute "
                        "WHERE attrelid=%s AND attnum>0 AND NOT attisdropped ORDER BY attnum", (oid,))
            resources.append({'schema': namespace, 'name': name, 'oid': oid,
                              'columns': [list(column) for column in cur]})
    return {'database': database, 'database_oid': database_oid, 'schema': schema,
            'server_address': address, 'server_port': port, 'tables': resources}
