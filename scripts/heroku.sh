#!/bin/bash

heroku container:push web --app=sadedegel --recursive
heroku container:release web --app=sadedegel
