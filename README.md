<h1 align='center'>Parking Spot Tracking</h1>
<p align='center'>Use AI to track parking time. Number of parking spaces</p>

https://user-images.githubusercontent.com/42292760/165655300-a8c3a4bf-97fc-464b-98c0-403f350315b1.mp4

# Requirements
- keras==2.8.0
- numpy==1.22.3
- opencv-python==4.5.5.64
- scipy==1.8.0
- tensorflow==2.8.0

# How to run:
- create virtualenv and activate
```
$ python3 -m vỉrtualenv venv
$ source venv/bin/activate
```

- install all package in requirements.txt
```
(venv) $ pip install -r requirements.txt
```

- run main.py script
```
(venv) $ python main.py
```

# Other dataset:
## Step 1: Marking parking area positions
![Screenshot from 2022-04-23 02-07-49](https://user-images.githubusercontent.com/42292760/164778727-d42b1d8b-9453-4fcb-a3b0-9c6bda6c248f.png)

## Step 2: Crop all parking slot image from frame of video to dataset
run script: ```ParkingAreaPosition2CSV.py```

## Step 3: Tranning model to classification beweent car nan non_car in parking slot
run script: ```trainning.py```

## Step 4: Predict
run script ```main.py```
