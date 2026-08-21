# PotatoSense AI

### Vision-Based Potato Moisture Content Estimation

PotatoSense AI is a computer vision and machine learning project for estimating the **moisture content of potato slices from images**.

The main idea behind the project is to explore whether moisture content can be estimated without using a dedicated moisture sensor or performing the conventional oven-drying method for every measurement.

The project uses a pretrained **ResNet50** model to extract features from potato-slice images. These features are then used with different regression models to predict the moisture content.

---

## About the Project

Moisture content is an important parameter during potato drying and processing. The commonly used oven-drying method is accurate but destructive and time-consuming. On the other hand, dedicated online moisture sensors can be expensive and require additional hardware.

This project looks at a different approach:

**Can we estimate the moisture content of a potato slice just from its image?**

To investigate this, images of potato slices were collected during infrared drying along with their corresponding moisture content measurements. Different image preprocessing methods, feature extraction techniques, and regression models were then tested.

The final system can take an image as input and return an estimated moisture content value.

---

## How It Works

The overall workflow is:

```text
Potato Slice Image
        ↓
Image Preprocessing
        ↓
ResNet50 Feature Extraction
        ↓
2048-D Feature Vector
        ↓
Regression Model
        ↓
Predicted Moisture Content (%)
```

There is also a separate **potato validity check** before prediction. This prevents the model from making a moisture prediction when the uploaded image does not look sufficiently similar to the potato-slice images used during development.

---

## Dataset

The images were collected while potato slices were undergoing infrared drying.

The experimental setup used:

* Potato slices of approximately 2.5 mm thickness
* Blanching at 90°C for 3 minutes
* Infrared drying
* One image approximately every 1.5 minutes
* A moisture-content measurement associated with each image

The dataset used during the study contains approximately **400 image–moisture-content pairs**, with moisture values covering roughly **0–86%**.

---

## Image Preprocessing

Before extracting features, the images go through several preprocessing steps.

### 1. ROI Extraction

The potato slice is separated from the background using thresholding and contour detection.

### 2. White Balance

A gray-world based correction is applied to reduce differences caused by lighting conditions.

### 3. Histogram Matching

Images are matched against a reference image to make their colour and exposure more consistent.

### 4. Denoising

A bilateral filter is used to reduce image noise while keeping important edges and texture.

### 5. Resize

The processed image is converted to **224 × 224 pixels**, which is the input size used for the ResNet50 feature extractor.

CLAHE was also tested during preprocessing, but it did not improve the results and was therefore not used in the final pipeline.

---

## Feature Extraction

Two types of features were investigated.

### ResNet50 Features

A pretrained ResNet50 model was used as a feature extractor rather than as a direct classifier.

The classification layer was removed and the output from the global average pooling layer was used as the image representation.

This gives a **2048-dimensional feature vector** for every image.

### GLCM + Colour Features

Handcrafted features were also extracted as a baseline.

These included:

**GLCM features**

* Contrast
* Homogeneity
* Energy
* Correlation
* ASM

**Colour features**

* BGR
* HSV
* LAB

This comparison helped determine whether the deep features from ResNet50 were actually useful for moisture prediction compared with more traditional image features.

---

## Regression Models

Several regression algorithms were tested using both feature types:

* Ridge Regression
* Lasso
* ElasticNet
* Linear SVR
* RBF SVR
* KNN
* Random Forest
* Gradient Boosting
* PLS
* PCA + Ridge
* PCA + SVR

The models were evaluated using:

* **MAE (Mean Absolute Error)**
* **R² Score**

The experiments showed that the **ResNet50-based features performed better overall than the handcrafted GLCM and colour features**.

---

## Potato Validity Check

One problem with an image-based prediction system is that a user could upload an image that has nothing to do with potatoes.

To handle this, the system includes a simple validity check.

The ResNet50 feature vector of the uploaded image is compared with stored potato-slice feature vectors using **cosine similarity**.

If the similarity is below the selected threshold, the image is rejected instead of sending it to the moisture prediction model.

```text
Uploaded Image
      ↓
ResNet50 Features
      ↓
Cosine Similarity
      ↓
 ┌───────────────┐
 │ Looks like    │
 │ potato slice? │
 └───────┬───────┘
         │
    Yes  │  No
         │
         ▼
   Prediction     Reject
```

This is mainly intended to avoid giving a meaningless moisture value for an invalid input.

---

## Deployment

The model was also deployed as a web application so that it can be used without setting up the complete ML environment locally.

The deployment setup is:

```text
Streamlit
    ↓
Docker
    ↓
AWS EC2
```

The Streamlit interface allows a user to upload an image and view the predicted moisture content.

The Docker container contains the application, model files, and required reference files so that the same setup can be run consistently.

---

## Technologies Used

### Python / Machine Learning

* Python
* NumPy
* Scikit-learn
* ResNet50
* Regression Models

### Image Processing

* OpenCV
* GLCM
* Colour-space features

### Deployment

* Streamlit
* Docker
* AWS EC2



## Results

The model experiments showed that features extracted from **ResNet50 were more useful for moisture prediction than the handcrafted GLCM and colour features**.

For deployment, the focus was not only on predictive performance but also on keeping the inference pipeline relatively lightweight so that it could run on a small cloud instance.

