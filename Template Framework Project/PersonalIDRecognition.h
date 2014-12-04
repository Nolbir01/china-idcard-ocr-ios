//
//  PersonalIDRecognition.h
//  Template Framework Project
//
//  Created by wl on 12/3/14.
//  Copyright (c) 2014 Daniele Galiotto - www.g8production.com. All rights reserved.
//

#ifndef __Template_Framework_Project__PersonalIDRecognition__
#define __Template_Framework_Project__PersonalIDRecognition__

#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

IplImage* LoadImage(const char *imagePath);
int recgonize(IplImage* iimg, const char *imagePath);

#endif /* defined(__Template_Framework_Project__PersonalIDRecognition__) */
