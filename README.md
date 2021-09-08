# critic-guided-segmentation-of-rewarding-objects-in-first-person-views
Critic Guided Segmentation of Rewarding Objects in First-Person Views

![Segmentation masks learned from sparse reward signal image](imgs/results.png)
Segmentation masks learned from sparse reward signal

## How to run our model
Train the model and produce an evaluation video:
`python main.py -train`

Convert your custom images with the trained model. They need to be stored as 64x64 pixel rgb images in a single folder. Default output folder is called `results`:
`python main.py --input-folder INPUTFOLDERNAME [--output-folder OUTPUTFOLDERNAME]`
