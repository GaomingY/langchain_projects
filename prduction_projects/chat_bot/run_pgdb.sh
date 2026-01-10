docker run -d \
    --name my-postgres \
    --restart always \
    -e POSTGRES_USER=ygm \
    -e POSTGRES_PASSWORD=ygm123456 \
    -e POSTGRES_DB=postgres \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    postgres:15