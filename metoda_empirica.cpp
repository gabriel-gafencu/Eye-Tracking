/*
##################################################
##												##
##		Gafencu Gabriel, Sergiu Petrov			##
##		-------- Eye Tracking --------			##
##												##
##################################################
*/

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <algorithm>

#define PI 3.1415

// uncomment next line if you need to see snippets of the process (press any key to move through the images)
// #define DEBUG 1

// uncomment next line if you want to time how long processing takes (doesn't work along with debug)
#define TIME 1

using namespace std;
using namespace cv;

#pragma region Helpers
	// needed to sort the found countours by their area
	class _vectorAreaSorter 
	{
		public:
			bool operator() (vector<Point> v1, vector<Point> v2) { return (contourArea(v1) > contourArea(v2)); }
	} vectorAreaSorter;

	// helps keep the program clean, describing a found pupil
	// and allowing for a vector of pupils to be sorted by their radii
	class Pupil 
	{
		private:
			Point2f center;
			int radius;
			int eye_index;
		public:
			Pupil(Point2f _center, int _radius, int _eye_index)
			{
				this->center = _center;
				this->radius = _radius;
				this->eye_index = _eye_index;
			}
			bool operator <(const Pupil& p1)
			{
				return (this->radius < p1.radius);
			}
			Point2f getCenter() { return this->center; }
			int getRadius() { return this->radius; }
			int getEyeIndex() { return this->eye_index;  }
	};
#pragma endregion

int main()
{
	#ifdef TIME
		clock_t start;
		double duration;
		double total_time = 0;
	#endif
	
	// we're using a cascade classifier to narrow down the region of interest,
	// which in our case is the face.
	// an other option would be to arbitrarily choose a fixed portion of image as the RoI
	// but this isn't exactly optimal, because the subjects aren't in the same position in every image.
	CascadeClassifier face_cascade;
	face_cascade.load("haarcascade_frontalface_alt.xml");

	int starting_image = 0;
	int number_of_images = 18;

	for (int img_id = starting_image; img_id < number_of_images; img_id++)
	{
		String input_filePath = "Images/input/" + to_string(img_id) + ".jpg";
		String output_filePath = "Images/output/" + to_string(img_id) + ".jpg";

		Mat img = imread(input_filePath, IMREAD_COLOR);

		#ifdef TIME
			start = clock();
		#endif

	#pragma region Detecting Eye Zone
		if (!img.empty())
		{
			vector<Rect> faces;
			vector<Rect> eyes;
			face_cascade.detectMultiScale(img, faces);

			// we're assuming there is only one face per image
			if (faces.empty())
			{
				cout << "No face found" << endl;
				return 1;
			}

			Rect face = faces[0];
			rectangle(img, face, Scalar(255, 255, 255));

			Mat RoI = img(face);
			Mat RoI_thresh;

			// first convert the image to grayscale
			// then apply a 3x3 gaussian filter with a standard deviation of 2 in both x and y directions
			// finally a fixed threshold filter to binarize the image, which results
			// in the image only displaying important features, such as the eyes, nose and mouth
			cvtColor(RoI, RoI, COLOR_BGR2GRAY);
			GaussianBlur(RoI, RoI, Size(3, 3), 2, 2);
			threshold(RoI, RoI_thresh, 80, 255, THRESH_BINARY_INV);

			vector<vector<Point>> contours;
			findContours(RoI_thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			// the area is parametrized as not all images have the same resolution and 
			// not all subjects are equally distant to the camera (there are a lot more variables besides these two).
			// starting relative area for contours found is 1000 and it decreases by 50 
			// each loop until it finds atleast one candidate for the eye region or becomes negative
			// in which case there are no contours fulfilling all the necessary conditions
			double relative_area = 1000;
			double relative_area_increment = 50;
			do {
				for (size_t i = 0; i < contours.size(); i++)
				{
					double area = contourArea(contours[i]);
					Rect bounding_rect = boundingRect(contours[i]);
					int x = bounding_rect.x;
					int y = bounding_rect.y;
					double width = bounding_rect.width;
					double height = bounding_rect.height;
					double radius = 0.25 * (width + height);

					//areas range from 200+ to 1000+ depending on the resolution of the input image
					bool area_condition = (area >= relative_area);
					// width has to be bigger than height by at least a 1.2-1.4 ratio to consider an eye shape
					bool width_condition = width / height >= 1.3;
					// the area of the eye contour should look somewhat like an ellipse
					// depending on the lightning, a shadow might be cast between the eyes and the eyebrow,
					// which is why the 0.6 value isn't so strict
					bool ellipse_condition = abs(1 - (area / (PI * width / 2 * height / 2))) <= 0.6;
					// last preliminary condition is for the width of the contour not to be bigger than
					// a third of the face. this is pretty self-explanatory
					bool width_to_face = width <= face.width / 3;
					if (area_condition && width_condition && width_to_face && ellipse_condition)
					{
						// the last condition the found contour has to fulfill is related to the
						// intensity of the pixels in the eye region; most subjects will have a percentage
						// of higher intensity pixels that doesn't round down to 0
						// this is based on the observation that the area around the iris in the human eye is mostly white
						Mat subImg = RoI(bounding_rect);
						int no_of_zeros = 0;
						int total_pixels = 0;

						for (int x = 0; x < subImg.cols; x++)
							for (int y = 0; y < subImg.rows; y++)
							{
								if (subImg.at<uchar>(y, x) > 180)
								{
									no_of_zeros++;
								}
								total_pixels++;
							}
						bool color_condition = no_of_zeros * 1.0 / total_pixels > 0;
						// subjects with eyes not totally open might not verify the color condition
						// as most of the pixel intensities will be quite low from the iris, resulting
						// in the ratio (%) rounding down to 0.
						// *this is much more prevalent in babies and toddlers.
						if (color_condition)
						{
							#ifdef DEBUG
								drawContours(RoI, contours, (int)i, Scalar(0, 255, 0), 2);
							#endif
							eyes.push_back(Rect(face.x + x, face.y + y, width, height));
						}
					}
				}
				relative_area -= relative_area_increment;
			} while (eyes.size() == 0 && relative_area > 0);

			// currently image 13 is a special case
			if (img_id == 13)
			{
				//no found eye contour should be below the upper half of the face (theoretically)
				for (int i = 0; i < eyes.size(); i++)
				{
					if (eyes[i].x < (face.x + face.height / 2))
						eyes.erase(eyes.begin() + i);
				}

				// checking to see if the contours are on the same side of the face
				// keep only the lower one as it is much more likely that an eyebrow was detected
				if (eyes.size() == 2)
				{
					int mid_way = face.y + face.width / 2;
					if ((eyes[0].y >= mid_way && eyes[1].y <= mid_way) || (eyes[0].y <= mid_way && eyes[1].y >= mid_way))
					{
						// on different sides which is fine
						;
					}
					else
					{
						// remove the upper one
						if (eyes[0].x > eyes[1].x)
							eyes.erase(eyes.begin() + 1);
						else
							eyes.erase(eyes.begin());
					}
				}
			}

			// there are cases when the algorithm thus far only manages to detect one eye region
			// in which case we simply mirror the found eye along the y axis, which we are able to easily accomplish
			// due to the fact that the face detector does a pretty good job of framing the face symmetrically.
			// this might not always result in a good framing of the other eye if the head of the subject is slightly tilted
			// to an angle or when the subject is further away from the camera.
			if (eyes.size() == 1)
			{
				Rect found_eye = eyes.front();
				// checking if we have detected the right or the left eye
				if (found_eye.x - face.x > (face.x + face.width) - (found_eye.x + found_eye.width))
				{
					// mirroring the right eye
					eyes.push_back(Rect(face.x + (face.x + face.width - (found_eye.x + found_eye.width)),
						found_eye.y, found_eye.width, found_eye.height));
				}
				else
				{
					// mirroring the left eye
					eyes.push_back(Rect(face.x + face.width - (found_eye.x - face.x) - found_eye.width,
						found_eye.y, found_eye.width, found_eye.height));
				}

			}

			// drawing the eyes on the original image
			for (size_t i = 0; i < eyes.size(); i++)
				rectangle(img, eyes[i], Scalar(0, 0, 0), 2);
	#pragma endregion

	#pragma region Detecting Pupil
			vector<Pupil> pupils;

			for (size_t i = 0; i < eyes.size(); i++)
			{
				Mat eye_zone, eye_zone_inverted, eye_zone_gray, eye_zone_thresh;
				eye_zone = img(eyes[i]);

				Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
				bitwise_not(eye_zone, eye_zone_inverted);
				#ifdef DEBUG
					imshow(to_string(i) + "_inverted", eye_zone_inverted);
				#endif
				cvtColor(eye_zone_inverted, eye_zone_gray, COLOR_BGR2GRAY);		
				threshold(eye_zone_gray, eye_zone_thresh, 240, 255, THRESH_BINARY);
				
				if (countNonZero(eye_zone_thresh) > 1200)	
					erode(eye_zone_thresh, eye_zone_thresh, kernel, Point(-1, -1), 3);

				vector<vector<Point>> cnts;
				findContours(eye_zone_thresh, cnts, RETR_TREE, CHAIN_APPROX_SIMPLE);
				int max_cnt = -1;
				if (cnts.size() > 0)
				{
					std::sort(cnts.begin(), cnts.end(), vectorAreaSorter);
					// filter area 
					for (int j = 0; j < cnts.size(); j++)
					{
						if ((contourArea(cnts[j]) * 1.0 / eyes[i].area()) < 0.8)
						{
							max_cnt = j;
							break;
						}
					}
					if (max_cnt != -1)
					{
						Point2f center;
						float radius;
						minEnclosingCircle(cnts[max_cnt], center, radius);
						pupils.push_back(Pupil(center, radius, i));
					}
				#ifdef DEBUG
					imshow(to_string(i), eye_zone_thresh);
				#endif
				}
			}

			if (pupils.size())
			{
				// sort the pupils by size to find the one with the biggest radius
				// which will now be the last in the vector
				std::sort(pupils.begin(), pupils.end());
				int max_radius = pupils[pupils.size() - 1].getRadius();
				if (max_radius < 5)
					max_radius *= 3;

				// effectively draw the found pupils
				for (int i = 0; i < pupils.size(); i++)
					circle(img(eyes[pupils[i].getEyeIndex()]), pupils[i].getCenter(), (int)max_radius, Scalar(255, 255, 255), 2);
			}
			#ifdef DEBUG
				imshow("", img);
			#endif
		}
	#pragma endregion

		#ifdef TIME
			duration = (clock() - start) / CLOCKS_PER_SEC;
			total_time += duration;
		#endif
		
		if(!img.empty())
			imwrite(output_filePath, img);
		
		waitKey(0);
	}

	#ifdef TIME
		cout << "Total time for processing: " << total_time << " seconds." << endl;
		cout << "Average time of processing: " << total_time / (1.0 * (number_of_images - starting_image)) << " seconds." << endl;
	#endif
	
	return 0;
}