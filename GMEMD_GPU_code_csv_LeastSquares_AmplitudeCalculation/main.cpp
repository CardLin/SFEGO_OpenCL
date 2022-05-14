#include<stdio.h>
#include<string.h> 
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
#include<CL/cl.h>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define ar_size 1000000
#define P_size 30
#define MAX_SOURCE_SIZE (0x1000000)
#define M_PI 3.14159265358979323846

struct Circle_Data {
        int x,y;
		int weight;
		float radius;
        float deg;
        bool token;
        int value;
        //float value;
} ar[ar_size];
int ar_len;
int ardatax[ar_size],ardatay[ar_size];
float ardatadeg[ar_size];

int prime[P_size];

void search_prime(){
	int p=2,count=0,find;
	prime[count++]=p++;
	while(count<P_size){
		find=0;
		for(int i=0;i<count;++i){
			if(p%prime[i]==0){
				find=1;
			}
		}
		if(!find){
			prime[count++]=p++;
		}else ++p;
	}
}

int compare(const void * a, const void * b){
	if( ((Circle_Data *)a)->deg > ((Circle_Data *)b)->deg ) return 1;
	else if( ((Circle_Data *)a)->deg == ((Circle_Data *)b)->deg ){ //if same direct
		if( ((Circle_Data *)a)->radius > ((Circle_Data *)b)->radius ) return 1;
		else return -1;
	}
	else return -1;
}

void build_list(int radius){
	ar_len=0;
	for(int i=-radius;i<=radius;++i){
		for(int j=-radius;j<=radius;++j){
			//if( (int) (sqrt(i*i+j*j)+0.5) == radius ){
			//if( (int) (sqrt(i*i+j*j)+0.5) <= radius ){
			if(  ( sqrt(i*i+j*j) < (float)radius+1.0 ) && !(i==0&&j==0) ){
				ar[ar_len].x=j;
				ar[ar_len].y=i;
				ar[ar_len].deg=atan2(j,i);
				ar[ar_len].radius=sqrt(i*i+j*j);
				if(ar[ar_len].deg<0.0) ar[ar_len].deg+=M_PI*2;
				++ar_len;
				if(ar_len==ar_size){
					puts("ar_list_OOM!!");
					exit(1);
				}
			}
		}
	}
	qsort(ar,ar_len,sizeof(Circle_Data),compare);
	float t=1;
	for(int i=ar_len/2-1;i>=0;--i,t*=2){
		ar[i].weight=t;
	}
	t=-1;
	for(int i=ar_len/2;i<ar_len;++i,t*=2){
		ar[i].weight=t; 
	}
}

/*
void normalize_img(float *data,int width,int height,float target){
	float min=100000000.0,max=-10000000.0;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			if(data[j*width+i]>max) max=data[j*width+i];
			if(data[j*width+i]<min) min=data[j*width+i];
		}
	}
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			data[j*width+i]-=min;
			data[j*width+i]/=(max-min)/target;
		}
	}
}
*/

void normalize_img(float *data,int width,int height,float min_target,float max_target,float &min,float &max,float &amplitude){
	min=100000000.0,max=-10000000.0;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			if(data[j*width+i]>max) max=data[j*width+i];
			if(data[j*width+i]<min) min=data[j*width+i];
		}
	}
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			data[j*width+i]-=min;
			data[j*width+i]/=(max-min)/(max_target-min_target);
			data[j*width+i]+=min_target;
		}
	}
	amplitude=max-min;
}

void subtraction_img(float *data1,float *data2,float *cut,int width,int height,float ratio){
	
	float *temp_data= (float*) malloc(sizeof(float)*width*height);
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			temp_data[j*width+i]=data2[j*width+i]*ratio;
		}
	}
	
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
				cut[j*width+i]=data1[j*width+i]-temp_data[j*width+i];
		}
	}
}

float correlation_coefficient(float *orig_data,float *target_data,int width,int height,float ratio){
	
	float *temp_data= (float*) malloc(sizeof(float)*width*height);
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			temp_data[j*width+i]=target_data[j*width+i]*ratio;
		}
	}
	
	//puts("");
	float avg_orig=0.0,avg_temp=0.0;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			//printf("%f,%f\n",orig_data[j*width+i],temp_data[j*width+i]);
			avg_orig+=orig_data[j*width+i];
			avg_temp+=temp_data[j*width+i];
		}
	}
	avg_orig/=(float)(width*height);
	avg_temp/=(float)(width*height);
	//printf("avg1=%f,avg2=%f,len=%f\n",avg_orig,avg_temp,(float)(width*height));
	
	float stdev_orig=0.0,stdev_temp=0.0,temp;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			temp=orig_data[j*width+i]-avg_orig;
			stdev_orig+=temp*temp;
			temp=temp_data[j*width+i]-avg_temp;
			stdev_temp+=temp*temp;
		}
	}
	stdev_orig=sqrt(stdev_orig);
	stdev_orig/=(float)(width*height-1.0);
	stdev_temp=sqrt(stdev_temp);
	stdev_temp/=(float)(width*height-1.0);
	//printf("stdev1=%f,stdev2=%f\n",stdev_orig,stdev_temp);
	
	
	float cov=0.0;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			cov+=(orig_data[j*width+i]-avg_orig)*(temp_data[j*width+i]-avg_temp);
		}
	}
	cov/=(float)(width*height-1.0);
	//printf("cov=%f\n",cov);
	
	float correl;
	correl=cov/(stdev_orig*stdev_temp);
	//printf("correl=%f\n",correl);
	
	float coeff;
	coeff=correl/(float)(width*height-1.0);
	//printf("coeff=%f\n",coeff);
	
	free(temp_data);
	return coeff;
}



void lbp_grad(unsigned char *img_data,float *lbp_diff,float *lbp_direct,int height,int width){
	int tx,ty;
	int pos,neg;
	float pos_avg,neg_avg;
	int pos_count,neg_count,weight,max_token;
	int now_data,target_data;
	int start,mid,end;
	//calc diff and direct
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			pos=0; neg=0; pos_count=0; neg_count=0;
			now_data=img_data[j*width+i];
			//calc difference
			for(int k=0;k<ar_len;++k){
				tx=j+ar[k].x;
				ty=i+ar[k].y;
				if( (tx>=0 && tx<height) && (ty>=0 && ty<width) ){
					target_data=img_data[tx*width+ty];
					ar[k].value=target_data;
					if(target_data>now_data){ //1
						//direct+=ar[k].weight;
						pos+=(int)target_data;
						++pos_count;
						ar[k].token=true; //required by direction
					}
					else{ //0
						neg+=(int)target_data;
						++neg_count;
						ar[k].token=false; //required by direction
					}
				}
			}
			
			//calc diff 
			if(pos_count) pos_avg=(float)pos/pos_count;
			else pos_avg=(float)now_data;
			if(neg_count) neg_avg=(float)neg/neg_count;
			else neg_avg=(float)now_data;
			lbp_diff[j*width+i]=pos_avg-neg_avg;
			
			//calc direction
			for(int k=0,l=ar_len/2;k<ar_len/2;++k,++l){
				if(ar[k].token && ar[l].token){
					if(ar[k].value>ar[l].value) ar[l].token=0;
					else ar[k].token=0;
				}
			}
			start=0;
			end=ar_len/2;
			mid=end/2;
			weight=0;
			for(int k=start;k<=end;++k){
				if(ar[k].token) ++weight;
			}
			ar[mid].weight=weight;
			for(int k=1;k<ar_len;++k){
				start=(start+1)%ar_len;
				end=(end+1)%ar_len;
				mid=(mid+1)%ar_len;
				if(ar[start].token) --weight;
				if(ar[end].token) ++weight;
				ar[mid].weight=weight;
			}
			max_token=0; weight=ar[0].weight;
			for(int k=1;k<ar_len;++k){
				if(ar[k].weight > weight){
					max_token=k;
					weight=ar[k].weight;
				}
				else if(ar[k].weight == weight){
					if(ar[k].value > ar[max_token].value){
						max_token=k;
						weight=ar[k].weight;
					}
				}
			}
			lbp_direct[j*width+i]=ar[max_token].deg;
		}
	}
}

void img_gaussian(float *lbp_diff,float *lbp_diff_gauss,int height,int width){
	int tx,ty;
	float sum,avg;
	int count;
	for(int i=0;i<width;++i){
		for(int j=0;j<height;++j){
			sum=lbp_diff[j*width+i]; count=1;
			//calc difference
			for(int k=0;k<ar_len;++k){
				tx=j+ar[k].x;
				ty=i+ar[k].y;
				if( (tx>=0 && tx<height) && (ty>=0 && ty<width) ){
					sum+=lbp_diff[tx*width+ty];
					++count;
				}
			}
			lbp_diff_gauss[j*width+i]=sum/count;
		}
	}
}

void integral_imf(float *imf_image,float *lbp_diff,float *lbp_direct,int height,int width){
	int tx,ty;
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			for(int k=0;k<ar_len;++k){
				tx=j+ar[k].x;
				ty=i+ar[k].y;
				if( (tx>=0 && tx<height) && (ty>=0 && ty<width) ){
					//imf_image[tx*width+ty]+=cos( ar[k].deg-lbp_direct[j*width+i] )*lbp_diff[j*width+i];
					imf_image[j*width+i]-=cos( lbp_direct[tx*width+ty]-ar[k].deg )*lbp_diff[tx*width+ty];
					//imf_image[j*width+i]+=cos( lbp_direct[tx*width+ty]-lbp_direct[j*width+i] )*lbp_diff[tx*width+ty];
				}
			}
		}
	}
}

	cl_platform_id platform_id[2];
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	cl_context context;
	cl_command_queue command_queue;
	cl_mem data_mem_obj;
	cl_mem diff_mem_obj;
	cl_mem direct_mem_obj;
	cl_mem result_mem_obj;
	cl_mem listx_mem_obj;
	cl_mem listy_mem_obj;
	cl_mem listdeg_mem_obj;
	cl_program program;
	cl_kernel kernel_gradient;
	cl_kernel kernel_integral;
	size_t log_size;
	size_t local_item_size[2] = {16,16};
	size_t global_item_size[2];
	cl_event event_gradient,event_integral;
	
void init_cl(int DATA_SIZE){
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen("kernel.cl", "r");
	if(!fp){
	fprintf(stderr,"Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	ret = clGetPlatformIDs(2, platform_id, &ret_num_platforms);
	//clGetPlatformInfo()
	ret = clGetDeviceIDs( platform_id[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	// Create an OpenCL context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	// Create memory buffers on the device for each vector 
	data_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE*sizeof(float), NULL, &ret);
	diff_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE*sizeof(float), NULL, &ret);
	direct_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE*sizeof(float), NULL, &ret);
	result_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE*sizeof(float), NULL, &ret);
	listx_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ar_size*sizeof(float), NULL, &ret);
	listy_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ar_size*sizeof(float), NULL, &ret);
	listdeg_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ar_size*sizeof(float), NULL, &ret);
	
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	//Build the program

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//Debugger!!

	char* build_log;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size+1];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	//puts(build_log);
	
	// Create the OpenCL kernel
	kernel_gradient = clCreateKernel(program, "GMEMD_gradient", &ret);
	kernel_integral = clCreateKernel(program, "GMEMD_integral", &ret);

	// Set the arguments of the kernel

}

void run_cl(float *data,float *result,float *diff,float *direct,int width,int height,int r){
	int DATA_SIZE=width*height;

	// Copy host data to memory buffers
	ret = clEnqueueWriteBuffer(command_queue, data_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), data, 0, NULL, NULL);
	//ret = clEnqueueWriteBuffer(command_queue, diff_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), hy, 0, NULL, NULL);
	//ret = clEnqueueWriteBuffer(command_queue, direct_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), ex, 0, NULL, NULL);
	//ret = clEnqueueWriteBuffer(command_queue, result_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), hy, 0, NULL, NULL);
	// Create a program from the kernel source
	
	build_list(r);
	for(int i=0;i<ar_len;++i){
		ardatax[i]=ar[i].x;
	}
	ret = clEnqueueWriteBuffer(command_queue, listx_mem_obj, CL_TRUE, 0, ar_len * sizeof(int), ardatax, 0, NULL, NULL);
	for(int i=0;i<ar_len;++i){
		ardatay[i]=ar[i].y;
	}
	ret = clEnqueueWriteBuffer(command_queue, listy_mem_obj, CL_TRUE, 0, ar_len * sizeof(int), ardatay, 0, NULL, NULL);
	for(int i=0;i<ar_len;++i){
		ardatadeg[i]=ar[i].deg;
	}
	ret = clEnqueueWriteBuffer(command_queue, listdeg_mem_obj, CL_TRUE, 0, ar_len * sizeof(float), ardatadeg, 0, NULL, NULL);
	
	ret = clSetKernelArg(kernel_gradient, 0, sizeof(cl_mem), &data_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 1, sizeof(cl_mem), &diff_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 2, sizeof(cl_mem), &direct_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 3, sizeof(cl_mem), &listx_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 4, sizeof(cl_mem), &listy_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 5, sizeof(cl_mem), &listdeg_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 6, sizeof(int), &ar_len);
	ret = clSetKernelArg(kernel_gradient, 7, sizeof(int), &width);
	ret = clSetKernelArg(kernel_gradient, 8, sizeof(int), &height);
	
	ret = clSetKernelArg(kernel_integral, 0, sizeof(cl_mem), &result_mem_obj);
	ret = clSetKernelArg(kernel_integral, 1, sizeof(cl_mem), &diff_mem_obj);
	ret = clSetKernelArg(kernel_integral, 2, sizeof(cl_mem), &direct_mem_obj);
	ret = clSetKernelArg(kernel_integral, 3, sizeof(cl_mem), &listx_mem_obj);
	ret = clSetKernelArg(kernel_integral, 4, sizeof(cl_mem), &listy_mem_obj);
	ret = clSetKernelArg(kernel_integral, 5, sizeof(cl_mem), &listdeg_mem_obj);
	ret = clSetKernelArg(kernel_integral, 6, sizeof(int), &ar_len);
	ret = clSetKernelArg(kernel_integral, 7, sizeof(int), &width);
	ret = clSetKernelArg(kernel_integral, 8, sizeof(int), &height);
	
	global_item_size[0]=width % local_item_size[0] == 0 ? width : (width / local_item_size[0] + 1) * local_item_size[0];
	global_item_size[1]=height % local_item_size[1] == 0 ? height : (height / local_item_size[1] + 1) * local_item_size[1];
	//printf("local=%d,%d global=%d,%d\n",(int)local_item_size[0],(int)local_item_size[1],(int)global_item_size[0],(int)global_item_size[1]);
	
	//run gradient
	ret = clEnqueueNDRangeKernel(command_queue, kernel_gradient, 2, NULL, global_item_size, local_item_size, 0, NULL, &event_gradient);
	clWaitForEvents(1, &event_gradient);
	
	/* don't read diff and direct
	ret = clEnqueueReadBuffer(command_queue, diff_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), diff, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, direct_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), direct, 0, NULL, NULL);
	*/
	
	//run integral
	ret = clEnqueueNDRangeKernel(command_queue, kernel_integral, 2, NULL, global_item_size, local_item_size, 0, NULL, &event_integral);
	clWaitForEvents(1, &event_integral);
	ret = clEnqueueReadBuffer(command_queue, result_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), result, 0, NULL, NULL);

/*
	clReleaseContext(context);
	clReleaseCommandQueue(command_queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel_gradient);
	clReleaseKernel(kernel_integral);
	clReleaseEvent(event_gradient);
	clReleaseEvent(event_integral);
	clReleaseMemObject(data_mem_obj);
	clReleaseMemObject(diff_mem_obj);
	clReleaseMemObject(direct_mem_obj);
	clReleaseMemObject(result_mem_obj);
	clReleaseMemObject(listx_mem_obj);
	clReleaseMemObject(listy_mem_obj);
	clReleaseMemObject(listdeg_mem_obj);
*/
}


int main(int argc, char *argv[]) {
	clock_t start_time,end_time;
	float used_time = 0;
	int default_radius[1000],default_radius_len=0;
	float default_ratio[1000],ratio;
	//int default_residual[1000];
	int t_width,t_height;
	FILE *fp;
	if( (fp=fopen("default_radius","r")) == NULL){
		puts("No input File!!");
		exit(1);
	}
	while(!feof(fp)){
		fscanf(fp,"%f",&default_ratio[default_radius_len]);
		fscanf(fp,"%d",&default_radius[default_radius_len++]);
		if(default_radius[default_radius_len-1]<=0) default_radius_len--;
		//printf("%d %d\n",default_ratio[default_radius_len-1],default_radius[default_radius_len-1]);
	}
	fclose(fp);
	srand (time(NULL));
	//search_prime();
	printf("%s\n",argv[1]);
	char OrigFileName[255];
	char FileName[255];
	if(argc<2){
		puts("Please input a image!!");
		exit(1);
	}
	strcpy(OrigFileName,argv[1]);
	//load image on gray
	IplImage *img = cvLoadImage(OrigFileName,CV_LOAD_IMAGE_GRAYSCALE);
	int channel=img->nChannels;
	int depth=img->depth;
	int width=img->width;
	int height=img->height;
	IplImage *img2; //= cvCreateImage(cvSize(width,height),depth,channel); 	//downsize
	IplImage *img3= cvCreateImage(cvSize(width,height),depth,channel);		//upsize
	IplImage *img4= cvCreateImage(cvSize(width*2,height),depth,channel);
	
	IplImage *fimg1= cvCreateImage(cvSize(width,height),IPL_DEPTH_32F,channel);
	IplImage *fimg2;
	//IplImage **fimg_array=(IplImage**) malloc(sizeof(IplImage*)*default_radius_len);
	/*
	for(int i=0;i>default_radius_len;++i){
		fimg_array[i]=cvCreateImage(cvSize(width,height),IPL_DEPTH_32F,channel);
	}
	*/
	//get Image data
	unsigned char *img_data= (unsigned char*) malloc(sizeof(unsigned char)*width*height*channel*16);
	float *orig_data= (float*) malloc(sizeof(float)*width*height*16);
	float *GMEMD_diff= (float*) malloc(sizeof(float)*width*height*16);
	//float *GMEMD_diff_gauss= (float*) malloc(sizeof(float)*width*height);
	float *GMEMD_direct= (float*) malloc(sizeof(float)*width*height*16);
	float *imf_image= (float*) malloc(sizeof(float)*width*height*16);
	float *sub_data= (float*) malloc(sizeof(float)*width*height*16);
	//float *imf_count= (float*) malloc(sizeof(float)*width*height);
	float tmp;

	//float **bimf_data;
	float *input_data = new float[width*height];
	float *combine_data = new float[width*height];
	float *calc_combine_data = new float[width*height];
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			input_data[j*width+i]=(float)((unsigned char)img->imageData[j*img->widthStep+i]);
			combine_data[j*width+i]=0.0;
			calc_combine_data[j*width+i]=0.0;
		}
	}
	
	float *min,*max,*amplitude,*mask_ratio,*mask_surface;
	min = new float[default_radius_len+1];
	max = new float[default_radius_len+1];
	amplitude = new float[default_radius_len+1];
	mask_ratio = new float[default_radius_len+1];
	mask_surface = new float[default_radius_len+1];
	
	float calibration_amplitude,calibration_min,calibration_max;
	
	float temp_min,temp_max,temp_amplitude;
	
	normalize_img(input_data,width,height,0.0,1.0,min[0],max[0],amplitude[0]);
	
	float** bimf_data = new float*[default_radius_len];
	for(int i = 0; i < default_radius_len; ++i) bimf_data[i] = new float[width*height];
	/*
	bimf_data=(float **)malloc(default_radius_len*sizeof(float *));
	for(int k=0;k>default_radius_len;++k) bimf_data[k]= (float*) malloc(sizeof(float)*width*height);
	*/
	float sum,avg;

	sprintf(FileName,"%s_gray.png",OrigFileName);
	cvSaveImage(FileName,img);
	
	init_cl(width*height*16);

	for(int rr=0,r;rr<default_radius_len;++rr){
		start_time=clock();
		r=default_radius[rr];
		ratio=default_ratio[rr];
		t_width=(int)(width/ratio);
		t_height=(int)(height/ratio);
		
		build_list(r);
		
		mask_ratio[rr+1]=ratio;
		mask_surface[rr+1]=ar_len;
		
		printf("ratio=%f\tradius=%d\tlen=%d\t ",ratio,r,ar_len);
		//system("pause");
		img2=cvCreateImage(cvSize(t_width,t_height),depth,channel);
		fimg2= cvCreateImage(cvSize(t_width,t_height),IPL_DEPTH_32F,channel);
		cvResize( img, img2, CV_INTER_LINEAR );
		for(int i=0;i<img2->width;++i){
			for(int j=0;j<img2->height;++j){
				for(int k=0;k<img2->nChannels;++k){
					img_data[(j*img2->width+i)*img2->nChannels+k]=img2->imageData[j*img2->widthStep+i*img2->nChannels+k];
					orig_data[j*img2->width+i]=(float)img_data[(j*img2->width+i)*img2->nChannels+k];
				}
			}
		}
		
		normalize_img(orig_data,t_width,t_height,0.0,1.0,min[0],max[0],amplitude[0]);
		
		run_cl(orig_data,imf_image,GMEMD_diff,GMEMD_direct,t_width,t_height,r);
		
		normalize_img(imf_image,t_width,t_height,0.0,1.0,min[rr+1],max[rr+1],amplitude[rr+1]);
		
		printf("\nAmp=%f\tRatio=%f\tSurface=%f\t",amplitude[rr+1],mask_ratio[rr+1],mask_surface[rr+1]);
		
		calibration_amplitude 	=	amplitude[rr+1] / (float)mask_surface[rr+1] ;
		calibration_min			=	min[rr+1] * (calibration_amplitude/amplitude[rr+1]);
		calibration_max			=	max[rr+1] * (calibration_amplitude/amplitude[rr+1]);
		amplitude[rr+1]		=	calibration_amplitude;
		min[rr+1]			=	calibration_min;
		max[rr+1]			=	calibration_max;

		/*
		for(int j=0;j<t_height-1;++j){
			for(int i=0;i<t_width-1;++i){
				cvSetReal2D(fimg2,j,i,imf_image[j*t_width+i]);
			}
		}
		cvResize( fimg2, fimg1, CV_INTER_LINEAR );
		//fimg_array[rr] = cvCloneImage(fimg1);
		//cvCopy(fimg1,fimg_array[rr]);
		
		sprintf(FileName,"%s_GMEMD_BIMF%.2f(%.2fx%d).csv",OrigFileName,ratio*r,ratio,r);
		if( (fp=fopen(FileName,"w"))==NULL ){
			printf("Open File:%s Error!!\n",FileName);
			system("pause");
			exit(1);
		}
		sum=0.0;
		for(int j=0;j<height;++j){
			//printf("%d\n",j);
			for(int i=0;i<width;++i){
				fprintf(fp,"%f,",cvGetReal2D(fimg1,j,i));
				//bimf_data[rr][j*width+i]=cvGetReal2D(fimg1,j,i);
				//printf("%d %d %f\n",j,i,bimf_data[rr][j*width+i]);
				//sum+=bimf_data[rr][j*width+i];
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
		*/
		
		//system("pause");
		
		/*
		avg=sum/(height*width);
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				bimf_data[rr][j*width+i]-=avg;
			}
		}
		*/
	
		//normalize_img(imf_image,t_width,t_height,1.0);
		for(int j=0;j<t_height;++j){
			for(int i=0;i<t_width;++i){
				img2->imageData[j*img2->widthStep+i]=(unsigned char)(imf_image[j*t_width+i]*255);
				//bimf_data[rr][j*width+i]=imf_image[j*t_width+i];
				cvSetReal2D(fimg2,j,i,imf_image[j*t_width+i]);
			}
		}
		sprintf(FileName,"%s_GMEMD_SpatialFrame%.2f(%.2fx%d).png",OrigFileName,ratio*r,ratio,r);
		cvResize( img2, img3, CV_INTER_LINEAR );
		cvSaveImage(FileName,img3);
		
		cvResize( fimg2, fimg1, CV_INTER_LINEAR );
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				bimf_data[rr][j*width+i]=cvGetReal2D(fimg1,j,i);
			}
		}
		normalize_img(bimf_data[rr],width,height,0.0,1.0,temp_min,temp_max,temp_amplitude);
		
		cvReleaseImage( &img2 );
		cvReleaseImage( &fimg2 );
		
		end_time=clock();
		used_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;
		printf("time=%f",used_time);
		puts("	done");
	}
	
	/*
	float **bimf_data=(float**) malloc(sizeof(float*)*default_radius_len);
	for(int k=0;k>default_radius_len;++k) bimf_data[k]= (float*) malloc(sizeof(float)*width*height);
	float sum,avg;
	
	//Normalized to zero
	for(int k=1;k<default_radius_len;++k){
		sum=0.0;
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				bimf_data[k][j*width+i]=cvGetReal2D(fimg_array[k],j,i);
				printf("%d %d %f\n",j,i,bimf_data[k][j*width+i]);
				sum+=bimf_data[k][j*width+i];
			}
		}
		avg=sum/(width*height);
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				bimf_data[k][j*width+i]-=avg;
			}
		}
	}
	*/
	
	/*
	//down sample parameter
	int downsample_size=7;
	int width_chunk_size=width/(downsample_size+1);
	int height_chunk_size=height/(downsample_size+1);
	
	//matrix AX=B, solve X
	MatrixXf A(downsample_size*downsample_size,default_radius_len+1);
	VectorXf B(downsample_size*downsample_size);
	VectorXf X(default_radius_len+1);
		
	//copy to matrix
	for(int k=0;k<default_radius_len;++k){
		for(int j=1,jj;j<height-1;++j){
			if(j%height_chunk_size==0){
				jj=(j/height_chunk_size)-1;
				for(int i=1,ii;i<width-1;++i){
					if(i%width_chunk_size==0){
						ii=(i/width_chunk_size)-1;
						//printf("%d %d %d %f\n",k+1,jj,ii,bimf_data[k][j*width+i]);
						A(ii*jj,k)=bimf_data[k][j*width+i];
						//if( A(ii*jj,k) != bimf_data[k][j*width+i]) printf("bug@(%d,%d)\n",ii,jj);
						printf("%d %d %d %f\n",k+1,jj,ii,A(ii*jj,k));
						if(k==0) B(ii*jj)=input_data[j*width+i];
					}
				}	
			}
		}
	}
	*/
	
	MatrixXf A(width*height,default_radius_len+1);
	VectorXf B(width*height);
	VectorXf X(default_radius_len+1);
	for(int k=0;k<default_radius_len;++k){
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				A(j*width+i,k)=bimf_data[k][j*width+i];
				if(k==0) B(j*width+i)=input_data[j*width+i];
			}
		}
	}
	for(int i=0;i<width*height;++i){ A(i,default_radius_len)=1.0; }
	//cout<<"A:\n"<<A<<endl;
	//cout<<"B:\n"<<B<<endl;
	
	//least squares with SVD
	X = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(B);
	
	//least squares with QR
	//X = A.colPivHouseholderQr().solve(B);
	
	cout<<"X:\n"<<X<<endl;
	
	sprintf(FileName,"%s_GMEMD_Amplitude.csv",OrigFileName);
	if( (fp=fopen(FileName,"w"))==NULL ){
		printf("Open File:%s Error!!\n",FileName);
		system("pause");
		exit(1);
	}
	fprintf(fp,"FileName,LeastSquareAmplitude,CalcMin,CalcMax,CalcAmplitude\n");
	for(int rr=0,r;rr<default_radius_len;++rr){
		r=default_radius[rr];
		ratio=default_ratio[rr];
		fprintf(fp,"%s_GMEMD_SpatialFrame%.2f(%.2fx%d).png,",OrigFileName,ratio*r,ratio,r);
		fprintf(fp,"%f,%f,%f,%f\n",X(rr),min[rr+1],max[rr+1],amplitude[rr+1]);
	}
	fprintf(fp,"%s_GMEMD_Constant.png,",OrigFileName);
	fprintf(fp,"%f\n",X(default_radius_len));
	fclose(fp);
	
	for(int rr=0,r;rr<default_radius_len;++rr){
		r=default_radius[rr];
		ratio=default_ratio[rr];
		sprintf(FileName,"%s_GMEMD_SpatialFrame%.2f(%.2fx%d).csv",OrigFileName,ratio*r,ratio,r);
		if( (fp=fopen(FileName,"w"))==NULL ){
			printf("Open File:%s Error!!\n",FileName);
			system("pause");
			exit(1);
		}
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				fprintf(fp,"%d,",(unsigned int)(bimf_data[rr][j*width+i]*X(rr)*65535.0));
				combine_data[j*width+i]+=bimf_data[rr][j*width+i]*X(rr);
				calc_combine_data[j*width+i]+=bimf_data[rr][j*width+i]*amplitude[rr+1];
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
	}
	
	//constant csv
	sprintf(FileName,"%s_GMEMD_Constant.csv",OrigFileName);
	if( (fp=fopen(FileName,"w"))==NULL ){
		printf("Open File:%s Error!!\n",FileName);
		system("pause");
		exit(1);
	}
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			fprintf(fp,"%d,",(unsigned int)(X(default_radius_len)*65535.0));
			combine_data[j*width+i]+=X(default_radius_len);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	//constant png
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			img3->imageData[j*img3->widthStep+i]=(unsigned char)(255);
		}
	}
	sprintf(FileName,"%s_GMEMD_Constant.png",OrigFileName);
	cvSaveImage(FileName,img3);
	
	//gray csv
	sprintf(FileName,"%s_gray.csv",OrigFileName);
	if( (fp=fopen(FileName,"w"))==NULL ){
		printf("Open File:%s Error!!\n",FileName);
		system("pause");
		exit(1);
	}
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			fprintf(fp,"%d,",(unsigned int)(input_data[j*width+i]*65535.0));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	//combine csv
	sprintf(FileName,"%s_GMEMD_Combine.csv",OrigFileName);
	if( (fp=fopen(FileName,"w"))==NULL ){
		printf("Open File:%s Error!!\n",FileName);
		system("pause");
		exit(1);
	}
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			fprintf(fp,"%d,",(unsigned int)(combine_data[j*width+i]*65535.0));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	//combine png
	normalize_img(combine_data,width,height,0.0,1.0,temp_min,temp_max,temp_amplitude);
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			img3->imageData[j*img3->widthStep+i]=(unsigned char)(combine_data[j*width+i]*255);
		}
	}
	sprintf(FileName,"%s_GMEMD_Combine.png",OrigFileName);
	cvSaveImage(FileName,img3);
	
	//calc combine csv
	sprintf(FileName,"%s_GMEMD_CalcCombine.csv",OrigFileName);
	if( (fp=fopen(FileName,"w"))==NULL ){
		printf("Open File:%s Error!!\n",FileName);
		system("pause");
		exit(1);
	}
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			fprintf(fp,"%d,",(unsigned int)(calc_combine_data[j*width+i]*65535.0));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	//calc combine png
	normalize_img(calc_combine_data,width,height,0.0,1.0,temp_min,temp_max,temp_amplitude);
	for(int j=0;j<height;++j){
		for(int i=0;i<width;++i){
			img3->imageData[j*img3->widthStep+i]=(unsigned char)(calc_combine_data[j*width+i]*255);
		}
	}
	sprintf(FileName,"%s_GMEMD_CalcCombine.png",OrigFileName);
	cvSaveImage(FileName,img3);
	
	system("pause");
	return 0;
}

