<h1>Satellite Imagery Segmentation Using UNet Architecture</h1>


<body>
<h3>License:</h3>
This repo contains a modified version of the code originally provided by xyzhang89 in his repository at:

(https://github.com/xyzhang89/DSTL_Image_Feature_Detection_Unet)

<h3>Motivation:</h3>
The purpose of this repo was to modify the code of xyzhang89 to be used with RGB channel images so to predict the model on images downloaded from google or other sources
Sample images should be copied to <code>samples/</code> directory, then refer to <code>predict.ipynb</code> to check the predictions


<h3>Libraries:</h3>
The following external libraries were used

<ul>
  <li>OpenCV</li>
  <li>tensorflow 2.0</li>
  <li>pandas</li>
  <li>tifffile</li>
  <li>Scikit Learn</li>
</ul>

<h3>Data:</h3>

To download the dataset, follow these steps

<ul>
  <li><code>!pip install kaggle</code></li>
  <li><code>mkdir data</code></li>
  <li><code>cd data</code></li>
  <li><code>touch /root/.kaggle/kaggle.json</code> for colab OR <code>touch ~/.kaggle/kaggle.json</code> for Linux </li>
  <li><code>echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_API_KEY"}' > "KAGGLE.JSON_PATH"</code></li>
  <li><code>kaggle competitions download -c dstl-satellite-imagery-feature-detection -f three_band.zip</code></li>
  <li><code>kaggle competitions download -c dstl-satellite-imagery-feature-detection -f grid_sizes.csv.zip</code></li>
  <li><code>kaggle competitions download -c dstl-satellite-imagery-feature-detection -f train_wkt_v4.csv.zip'</code></li>
</ul>

</body>

<h3>Train:</h3>

Use <code>python train.py</code> to train.

Weights will be saved in <code>weights/</code> with class_type included in their name based on the model you've trained

<h3>Predict:</h3>
Use <code>predict.ipynb</code> to predict and visualize

<h3>Output:</h3>

![Outputs](/test_visualization/test_outputs.png)
