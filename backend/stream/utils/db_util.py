from sqlalchemy import create_engine, MetaData, Table, Column, String


def create_table(table_name, engine):
    meta = MetaData()
    table = Table(
       table_name, meta,
       Column('url', String, primary_key=True),
       Column('title', String),
       Column('publishedAt', String),
       Column('addedAt', String)
    )
    meta.create_all(engine)
    return table


def update_article_url_db(article_dicts, table_name, db_file_path):
    engine = create_engine('sqlite:///{}'.format(db_file_path), echo=True)
    table = create_table(table_name, engine)
    with engine.connect() as connection:
        for article_dict in article_dicts:
            statement = table.update().values(article_dict)
            result = connection.execute(statement)
