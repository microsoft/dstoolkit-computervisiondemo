FROM python:3.9

ADD requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY flask_app flask_app

WORKDIR /flask_app

ENV PORT 8083

EXPOSE 8083

# When running our docker image using "docker run" this command gets run
# This will start our gunicorn web server and run our app
ENTRYPOINT ["gunicorn", "app:app"]
