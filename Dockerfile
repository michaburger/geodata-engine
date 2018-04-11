FROM python:3
ADD main.py /
ADD database.py /
ADD geometry.py / 
ADD heatmap.py / 
ADD mapping.py /
ADD ml_engine.py /
ADD plot.py /
ADD requirements.txt /
RUN pip install -r requirements.txt
CMD ["python3", "./main.py"]
