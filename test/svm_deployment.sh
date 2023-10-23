#!/bin/bash

echo
TXT="Hey Joe, I need you to book a trip to Antigua for the team."
echo $TXT

curl -d "[{\"text\": \"$TXT\"}]" \
     -H "Content-Type: application/json" \
     -X POST http://0.0.0.0:8000/predict
echo
TXT="Joe said the sun is bright."
echo $TXT

curl -d "[{\"text\": \"$TXT\"}]" \
     -H "Content-Type: application/json" \
     -X POST http://0.0.0.0:8000/predict

echo
TXT="Please, send me the file."
echo $TXT

curl -d "[{\"text\": \"$TXT\"}]" \
     -H "Content-Type: application/json" \
     -X POST http://0.0.0.0:8000/predict

echo
TXT="FYI. Can you make a call tomorrow at 2pm?"
echo $TXT

curl -d "[{\"text\": \"$TXT\"}]" \
     -H "Content-Type: application/json" \
     -X POST http://0.0.0.0:8000/predict

echo
TXT="He forgot the diamonds under the passenger seat"
echo $TXT
curl -d "[{\"text\": \"$TXT\"}]" \
     -H "Content-Type: application/json" \
     -X POST http://0.0.0.0:8000/predict

echo
# GET method info
curl -X GET http://localhost:8000/info

echo
# GET method health
curl -X GET http://localhost:8000/health
echo

