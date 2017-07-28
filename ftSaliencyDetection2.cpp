#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/core/core.hpp>  
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>  
#include <sstream>
#include <iomanip>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>  
#include "classification.hpp"

using namespace cv;
using namespace std;


std::vector<cv::Rect> fineMinAreaRect(Mat &threshold_output,Mat &org_image)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    std::vector<cv::Rect> detectedRect;
    //寻找轮廓
    findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    //对每个找到的轮廓创建可倾斜的边界框
    vector<RotatedRect> minRect(contours.size());
    vector<RotatedRect> orgRect(contours.size());
    float response[contours.size()];
    int     num=contours.size();
    int ROI_x_min[num],ROI_y_min[num],ROI_width[num],ROI_height[num];
    for (int i = 0; i < contours.size(); i++)
    {
        minRect[i] = minAreaRect(Mat(contours[i]));//Input vector of 2D points, stored in:
                                                   //std::vector<> or Mat (C++ interface)
        orgRect[i].center.x= minRect[i].center.x*4;
        orgRect[i].center.y = minRect[i].center.y*4;
        orgRect[i].size.height = minRect[i].size.height*8;
        orgRect[i].size.width = minRect[i].size.width*8;
        orgRect[i].angle = minRect[i].angle*0;
       // cout<<"contours["<<i<<"].size() "<<contours[i].size()<<"    minRect[i].center.x "<<minRect[i].center.x<<endl;

        ROI_x_min[i] = orgRect[i].center.x - orgRect[i].size.width/2 ;
        ROI_y_min[i] = orgRect[i].center.y - orgRect[i].size.width/2 ;
        ROI_x_min[i] = (ROI_x_min[i] > 0 ? ROI_x_min[i]:0);
        ROI_y_min[i] = (ROI_y_min[i] > 0 ? ROI_y_min[i]:0);
        ROI_width[i] = orgRect[i].size.width;
        ROI_height[i] = orgRect[i].size.height;
        
        ROI_width[i] = (ROI_width[i]+ROI_x_min[i] > (org_image.cols-1) ? (org_image.cols-1) - ROI_x_min[i]:ROI_width[i]);
        ROI_height[i] = (ROI_height[i]+ROI_y_min[i] > (org_image.rows-1)? (org_image.rows-1) - ROI_y_min[i]:ROI_height[i]);
        cout<<ROI_x_min[i]<<" "<<ROI_y_min[i] <<" "<<ROI_width[i]<<" "<<ROI_height[i]<<endl;
        

        if(ROI_width[i]<200&&ROI_height[i]<200&&ROI_width[i]>4&&ROI_height[i]>4&&ROI_x_min[i]>281)
        {   cv::Rect roiFound;
        	roiFound.x = ROI_x_min[i];
        	roiFound.y = ROI_y_min[i];
        	roiFound.width = ROI_width[i];
        	roiFound.height = ROI_height[i];

        	detectedRect.push_back(roiFound);
        }

    }
    
    return detectedRect;
    

}

void SaliencyProcess(Mat &I,Mat &invDFTcvt)
{
	if(I.empty())
		return;

	if(I.channels()==3)
	cvtColor(I,I,CV_RGB2GRAY);
	Mat planes[] = { Mat_<float>(I), Mat::zeros(I.size(), CV_32F) };
	Mat complexI; //复数矩阵
	merge(planes, 2, complexI); //把单通道矩阵组合成复数形式的双通道矩阵
	dft(complexI, complexI);  // 使用离散傅立叶变换

	//对复数矩阵进行处理，方法为谱残差
	Mat mag,pha,mag_mean;
	Mat Re,Im;
	split(complexI,planes); //分离复数到实部和虚部
	Re=planes[0]; //实部
	Im=planes[1]; //74
	magnitude(Re,Im,mag); //计算幅值
	phase(Re,Im,pha); //计算相角

	float *pre,*pim,*pm,*pp;
	//对幅值进行对数化
	for(int i=0;i<mag.rows;i++)
	{
		pm=mag.ptr<float>(i);
		for(int j=0;j<mag.cols;j++)
		{
  	 		*pm=log(*pm);
   	 		pm++;
		}
	}
	blur(mag, mag_mean, Size(5, 5)); //对数谱的均值滤波
	mag = mag - mag_mean; //求取对数频谱残差
	//把对数谱残差的幅值和相角划归到复数形式
	for(int i=0;i<mag.rows;i++)
	{
		pre=Re.ptr<float>(i);
		pim=Im.ptr<float>(i);
		pm=mag.ptr<float>(i);
		pp=pha.ptr<float>(i);
		for(int j=0;j<mag.cols;j++)
		{
		    *pm=exp(*pm);
		    *pre=*pm * cos(*pp);
		    *pim=*pm * sin(*pp);
		    pre++;
		    pim++;
		    pm++;
		    pp++;
		}
	}

	Mat planes1[] = { Mat_<float>(Re),Mat_<float>(Im) };

	merge(planes1, 2, complexI); //重新整合实部和虚部组成双通道形式的复数矩阵
	idft(complexI, complexI, DFT_SCALE); // 傅立叶反变换
	split(complexI, planes); //分离复数到实部和虚部
	Re=planes[0];
	Im=planes[1];
	magnitude(Re,Im,mag); //计算幅值和相角
	for(int i=0;i<mag.rows;i++)
	{
	    pm=mag.ptr<float>(i);
	    for(int j=0;j<mag.cols;j++)
	    {
		*pm=(*pm) * (*pm);
		pm++;
	    }
	}

	GaussianBlur(mag,mag,Size(7,7),2.5,2.5);

    Mat invDFT;
	normalize(mag,invDFT,0,255,NORM_MINMAX); //归一化到[0,255]供显示
	invDFT.convertTo(invDFTcvt, CV_8U); //转化成CV_8U型


}


string model_caffe = "./params/cifar10_quick.prototxt";
string para_caffe = "./params/hy_model_iter_320000.caffemodel";
string mean_caffe = "./params/cifar10_mean.binaryproto";
string label_caffe = "./params/label.txt";
int main(int argc,char *argv[])
{

    const char *videoname = (argc >= 2 ? argv[1] : "GOPR0591_jiequ.avi");
    
    ::google::InitGoogleLogging(argv[0]);

    const string& model_file   = model_caffe;
    const string& trained_file = para_caffe;
    const string& mean_file   = mean_caffe;
    const string& label_file = label_caffe;

    std::cout<<"##########################"<<std::endl;
    std::cout<<"model_file is   "<<model_file<<std::endl;
    std::cout<<"trained_file is   "<<trained_file<<std::endl;
    std::cout<<"##########################"<<std::endl;

  
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    VideoCapture capture(videoname);
    capture.set(CV_CAP_PROP_POS_FRAMES,10);/// play from 10 
    if (!capture.isOpened())
	{
		std::cout<< "No Input Image"<<std::endl;
		return 1;
	}

    namedWindow("org_image",CV_WINDOW_NORMAL);
    char Tex[20];
    char car1Tex[40];
    char car0Tex[40];
    VideoWriter writer("save1.avi", CV_FOURCC('M', 'J', 'P', 'G'), 20.0, Size(1280, 720));
    int N=0;

    while(1)
    {
        N=N+1;
        Mat I_org; // 当前视频帧
        if (!capture.read(I_org))
            break;

		Mat I,imOrg;
		I_org.copyTo(imOrg);

		struct timeval tv0,tv1;    
		gettimeofday(&tv0,NULL);    
		int startTime = int(tv0.tv_sec * 1000 + tv0.tv_usec / 1000);

		resize(I_org,I,Size(I_org.cols/4,I_org.rows/4));
        cout<<"I_org.cols "<<I_org.cols<<" I_org.rows"<<I_org.rows<<endl;
   	   
        Mat invDFTcvt,invDFTcvtSave;
		
		SaliencyProcess(I,invDFTcvt);
    
		threshold(invDFTcvt, invDFTcvt, 80, 255, CV_THRESH_BINARY);
	    invDFTcvt.copyTo(invDFTcvtSave);
	
        std::vector<cv::Rect> gDetectedRect = fineMinAreaRect(invDFTcvt,imOrg);/// 最小外接矩形

        string CAR0 = "0";
        string CAR1 = "1";
        float prob0 = 0;
        float prob1 = 0;
        int car0 = 10000;
        int car1 = 10000;
        for(int i = 0;i<gDetectedRect.size();i++)
        {   
        	Mat imageToClassify =  imOrg(gDetectedRect[i]);
        	std::cout<<"imageToClassify.cols "<<imageToClassify.cols<<"  imageToClassify.rows  "<<imageToClassify.rows<<std::endl; 
        	cv::imshow("imageToClassify",imageToClassify);
        	cv::waitKey(10);
            CHECK(!imageToClassify.empty()) << "Unable to decode image ";
            std::vector<Prediction> predictions = classifier.Classify(imageToClassify);//// how????
            /// Print the top N predictions. ///

              /* 
			for (size_t i = 0; i < predictions.size(); ++i) 
			{
			    Prediction p = predictions[i];
			    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			              << p.first << "\"" << std::endl;
			
                if((p.first == CAR0)&&(p.second>prob0))
                {
                	car0 = i;
                	prob0 = p.second;
                   
                }

                if((p.first == CAR1)&&(p.second>prob1))
                {
                	car1 = i;
                	prob1 = p.second;
                   
                }
            
			}
            */
            Prediction p = predictions[0];
        	if((p.first == CAR0)&&(p.second>prob0))
            {
            	car0 = i;
            	prob0 = p.second;
               
            }

            if((p.first == CAR1)&&(p.second>prob1))
            {
            	car1 = i;
            	prob1 = p.second;
               
            }

            rectangle(imOrg,gDetectedRect[i],Scalar(255,155,255),1,8,0);
        
   

        }

        if(prob0>0.2&&car0<10000)
        {
        	
        	if(car0==car1)
        	{
        		
        	}
        	else //if(prob0>prob1)
            {
            	sprintf(car0Tex,"car 0: %f",prob0);
        	    rectangle(imOrg,gDetectedRect[car0],Scalar(0,0,255),3,8,0);
        	    putText(imOrg,car0Tex,Point(gDetectedRect[car0].x,gDetectedRect[car0].y),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),4,8);//在图片上写文字
            }
            //else
            //{}
        	
        }

        if(prob1>0.2&&car1<10000)
        {
        	
        	 if(car0==car1)
        	{
        		
        	}
        	else //if(prob1>prob0)
            {
            	sprintf(car1Tex,"car 1: %f",prob1);
        		rectangle(imOrg,gDetectedRect[car1],Scalar(255,0,0),3,8,0);
        		putText(imOrg,car1Tex,Point(gDetectedRect[car1].x,gDetectedRect[car1].y),FONT_HERSHEY_SIMPLEX,1,Scalar(255,0,0),4,8);//在图片上写文字
            
            }
            //else
            //{}
        	
        }



	    gettimeofday(&tv1,NULL);    
	    int endTime = int(tv1.tv_sec * 1000 + tv1.tv_usec / 1000);
	    std::cout<<"endTime - startTime: "<<endTime - startTime<<" ms"<<std::endl;
        sprintf(Tex,"Fps: %d",int(1000/(endTime - startTime +1)));
        putText(imOrg,Tex,Point(50,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);//在图片上写文字

        writer.write(imOrg);
        cout<<imOrg.cols<<" "<<imOrg.rows<<endl;
        imshow("org_image",imOrg);  
        waitKey(3);    
      
    } 

    return 0;
}
