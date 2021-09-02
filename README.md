conda create --name autoFPS python=3.9.5
conda activate autoFPS
cd C:\dev\autoFPS



Install Visual C++ 2015 Build Tools from here with default selection.
https://go.microsoft.com/fwlink/?LinkId=691126

python detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img
python detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img  --person
python detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img  --heads