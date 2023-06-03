#!/bin/bash
docker stop ia; docker rm ia
( cd ia; docker build --rm -t ia . )
docker stop nn; docker rm nn
( cd nn; mvn install; docker build --rm -t nn . )

