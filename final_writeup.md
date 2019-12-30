## Final Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal_output/example_undistort.jpg "Undistorted Calibration image"
[image1]: ./output_images/straight_lines1-undistort.jpg "Undistorted Test image"
[image2]: ./output_images/test1-thresholded.jpg "Test1.jpg Thresholded"


[image3]: ./output_images/straight_lines1-perspective.jpg "Lane line perspective points"
[image4]: ./output_images/warped-compare-area-linear-interpolation.jpg "Example Warping (linear vs. area interpolation)"
[image5]: ./output_images/test1-windowed-pixel-search.jpg "Test1.jpg windowed pixel search"
[image6]: ./output_images/test1-windowed-polyfit.jpg "Test1.jpg windowed polynomial fit"
[image7]: ./output_images/test4-fit-search-prior.jpg "Test4.jpg Fit from Prior (test1.jpg)"

[image8]: ./output_images/test1-visualization.jpg "Test1.jpg with Lane Area, Curvature, and Vehicle offset"
[image9]: ./output_images/test4-visualization.jpg "Test4.jpg with Visualization"

[video1]: ./test_output_video/project_video_labeled-middle.mp4 "Difficult transition video"
[video2]: ./test_output_video/project_video-full.mp4 "Full Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell of the IPython notebook located in "P2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard (9x6) corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Undistortion example][image0]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![Undistorted straight_lines1.jpg][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (cell 7 in "P2.ipynb", function: `img2linebinary`).  Here's an example of my output for this step (note: test1.jpg was undistorted before this)

![Thresholded (undistorted) test1.jpg][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I use a PerspectiveTransform class to execute warping and unwarping of undistorted highway images in cell 6. `highway_birds_eye_view` warps an image such that a rhombus is transformed into a rectangle, and `car_view` method undoes this warping. `src` and `dst` are hardcoded - I chose these points visually from a sample undistorted image in cells 4 and 5:
![Visually selecting 'src' points from straight_lines1.jpg][image3]


```python
    dst = np.float32([[ 320.,  720.],
                      [ 320.,    0.],
                      [ 960.,    0.],
                      [ 960.,  720.]])
    src = np.float32([[  195.,   720.],
                      [  594.,   450.],
                      [  686.,   450.],
                      [ 1116.,   720.]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 195, 720      | 320, 720      | 
| 594, 450      | 320, 0        |
| 686, 450      | 960, 0        |
| 1116, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warping creates rectangle - linear vs. area interpolation][image4]

We notice that uncorrected camera distortion may contribute to the curvature at the lower base of the lane lines. This 
could be expected because camera distortion is stronger at the outer edges of an image, and camera calibration
may not have been able to perfectly estimate the distortion coefficients. Also the above image illustrates that "linear interpolation" is a good and faster approximation of the best interpolation method OpenCV offers.
This strategy of interpolation is what was used.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Overall, one of two strategies is used to 2 sets of lane line pixels - one set for the left and another for the right.
Next, `numpy.polyfit` was used to fit a quadratic to each line's pixels as a function of the pixel y-coordinate.

Cell 10 contains the implementation of "histogram windowed search" for lane pixels, modified from the quiz code, `find_lane_pixels_histogram_window`. Cell 15 contains the implementation of `find_lane_pixels_from_prior()`, modified from quiz code to use Line objects (which maintain lane line state such as latest fit coefficients).
Cell 11 contains the implementation of Line objects, which facilitates maintaining line state across multiple frames as well as
evaluating line characteristics like curvature, determining how valid are a frame's detected lines, etc.
A frame's detected fits are determined valid when they pass the following checks:
        - their real curvatures are reasonable (i.e. above minimum real curvature)
        - approximately are of reasonable width (at bottom of image)
        - are roughly parallel (checked at bottom and top of image)

The common input for these pixel search functions is a binary thresholded warped image. When searching using a prior fit, 2 Line objects 
are passed in to `find_lane_pixels_from_prior()` representing the best current estimate of the lane lines from previous frames.

Windowed-search begins by using a histogram to find left and right line starting x coordinates at the bottom of the input. 
14 windows are sequentially stacked as pixels are searched and counted within a 100px margin from the window below - new windows
will be recentered if more than 25 non-zero pixels are detected within a window below.

![Windowing used to search for pixels][image5]
![Polynomial fit from windowed-search][image6]

A "search using a prior fit" uses the best fit coefficients (i.e. average of last 10 good fits) of the input Line objects.
![Polynomial fit from pixels found around prior fit][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Cell 11 (at the bottom) contains the implementation of `compute_best_real_curvature` for a line. The reported curvature overlaid onto
the original image is the average of the left and right lines' best fits' curvature in meters. To convert units from pixels to meters, 
I chose to convert the coefficients themselves to be in meters before evaluating the curvature formula for a quadratic function.

Cell 11 also includes the implementation of a class method `compute_vehicle_lane_offset` that uses left and right Line objects to
compute the vehicle offset evaluated at the bottom of the image (i.e. closest to the vehicle). First, assuming the camera is centered over the car, the center of the car is the middle of the image (in pixels). 
Using the lane's two Line objects, we can determine x-coordinate (in pixels) where the center of the lane should be. The offset (converted to meters) is then the signed distance between car's center and the lane's center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Cell 23 implements the pipeline function, `detect_lane_boundaries`, to consume one frame at a time with the lane Line objects carried over from previous execution of this same function.
It returns the image overlaid with the lane area highlighted, the curvature reported in meters, and the vehicle offset in meters.

(`stateful_detect_lane_bounds` is a partial function wrapping `detect_lane_boundaries` that only accepts a single frame as input so it has a compatible signature for `VideoFileClip`'s `fl_image` per-frame-mapping method)

Below is an example output of the pipeline function.

![test1 with visualization][image8]
![test1 with visualization][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_output_video/project_video_full.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There were a lot of hyperparameters to tune for the pipeline, and the most brittle/sensistive were those used to threshold the images.
These were critical in order to ensure that enough pixels were found to fit accurately the lines under various conditions such as turns, shadowing, and asphalt color. The hyperparameters chosen are good at least for the most common image in the test video (black asphalt, no shadowing, shallow turns).

In order to handle the edge-case conditions of shadowing, lighter color asphalt, and turns, my strategy includes both smoothing and filtering in the processing pipeline. The "best fit" polynomial for each lane line is actually the average of the last 10 good fits, and the best fits are used as priors to search for line pixels. When lane lines detected are not trustworthy (according to the sanity checks discussed above), the best fit is still valid to be used to visualize the the lane area, curvature, and offset.
Filtering was also introduced: sanity checks (and an especially strict check that the two lane lines are parallel) ensure that bad frames don't ruin the running "best fit" polynomial tracking the true lane lines. If 4 consecutive frames are rejected, then the Lines are detected anew using the holistic "windowed-search" method, whose success is still sensitive to thresholding hyperparameters.

In the output video, we see that shadowing does not significantly impact the lane area detected (until the last shadow at 42s). This is thanks to the smoothing and filtering. However, the pipeline appears to briefly "flip" the lane area at the first transition from a lighter asphalt to the darker asphalt around 24s). The pipeline doesn't appear to find a good fit at 41s at the second similar transition.

To make the pipeline more robust to asphalt changes, we note that despite the changes in asphalt color, the lane curvature does not change much in these areas. In fact, curvature cannot change drastically from frame to frame because our test data is from a highway in which sharp turns are rarely seen. 
Additional filtering/sanity checks could be added to ensure that fits must not be too different from the running "best fit", which aligns well with this turn-sharpness assumption.

Overall, since this pipeline appears sensitive to lighting changes and turning assumptions relevant only to highways, I expect this pipeline to 
not work nearly as well processing video of a nighttime or non-highway scenarios. Also, highway scenarios with greater traffic could introduce a lot more noise to the selection of lane line pixels. For these scenarios, robustness could be improved by if hyperparameters were tuned to each scenario and logic could be added to determine the scenario at hand (and therefore which hyperparameter values that should be used). This is difficult to do by hand, which motivates introducing deep learning to segment out the 
highway lanes from the rest of the scene despite illumination conditions or road-type.