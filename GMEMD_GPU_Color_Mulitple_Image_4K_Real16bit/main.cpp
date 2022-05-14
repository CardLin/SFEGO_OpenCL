#include<stdio.h>
#include<string.h> 
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>
#include<CL/cl.h>
#define ar_size 1000000
#define P_size 30
#define MAX_SOURCE_SIZE (0x1000000)

struct Circle_Data {
        int x,y;
		int weight;
		float radius;
        float deg;
        bool token;
        int value;
        float ratio;
        //float value;
} ar[ar_size];
int ar_len;
int ardatax[ar_size],ardatay[ar_size];
float ardatadeg[ar_size],ardataratio[ar_size];

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
			if(  ( sqrt(i*i+j*j) <= (float)radius ) && !(i==0&&j==0) ){
				ar[ar_len].x=j;
				ar[ar_len].y=i;
				ar[ar_len].deg=atan2(j,i);
				ar[ar_len].radius=sqrt(i*i+j*j);
				ar[ar_len].ratio=sin(M_PI*(float)ar[ar_len].radius/(float)radius);
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

	cl_platform_id platform_id[3];
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
	cl_mem listratio_mem_obj;
	cl_program program=NULL;
	cl_kernel kernel_gradient;
	cl_kernel kernel_integral;
	size_t log_size;
	size_t local_item_size[2] = {16,16};
	size_t global_item_size[2];
	cl_event event_gradient,event_integral,event_readbuffer;
	
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

	ret = clGetPlatformIDs(3, platform_id, &ret_num_platforms);
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
	listratio_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ar_size*sizeof(float), NULL, &ret);
	
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
	if(program==NULL) init_cl(DATA_SIZE);
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
	
	for(int i=0;i<ar_len;++i){
		ardataratio[i]=ar[i].ratio;
	}
	ret = clEnqueueWriteBuffer(command_queue, listratio_mem_obj, CL_TRUE, 0, ar_len * sizeof(float), ardataratio, 0, NULL, NULL);
	
	ret = clSetKernelArg(kernel_gradient, 0, sizeof(cl_mem), &data_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 1, sizeof(cl_mem), &diff_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 2, sizeof(cl_mem), &direct_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 3, sizeof(cl_mem), &listx_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 4, sizeof(cl_mem), &listy_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 5, sizeof(cl_mem), &listdeg_mem_obj);
	ret = clSetKernelArg(kernel_gradient, 6, sizeof(int), &ar_len);
	ret = clSetKernelArg(kernel_gradient, 7, sizeof(int), &width);
	ret = clSetKernelArg(kernel_gradient, 8, sizeof(int), &height);
	//ret = clSetKernelArg(kernel_gradient, 9, sizeof(cl_mem), &listratio_mem_obj);
	
	ret = clSetKernelArg(kernel_integral, 0, sizeof(cl_mem), &result_mem_obj);
	ret = clSetKernelArg(kernel_integral, 1, sizeof(cl_mem), &diff_mem_obj);
	ret = clSetKernelArg(kernel_integral, 2, sizeof(cl_mem), &direct_mem_obj);
	ret = clSetKernelArg(kernel_integral, 3, sizeof(cl_mem), &listx_mem_obj);
	ret = clSetKernelArg(kernel_integral, 4, sizeof(cl_mem), &listy_mem_obj);
	ret = clSetKernelArg(kernel_integral, 5, sizeof(cl_mem), &listdeg_mem_obj);
	ret = clSetKernelArg(kernel_integral, 6, sizeof(int), &ar_len);
	ret = clSetKernelArg(kernel_integral, 7, sizeof(int), &width);
	ret = clSetKernelArg(kernel_integral, 8, sizeof(int), &height);
	ret = clSetKernelArg(kernel_gradient, 9, sizeof(cl_mem), &listratio_mem_obj);
	
	global_item_size[0]=width % local_item_size[0] == 0 ? width : (width / local_item_size[0] + 1) * local_item_size[0];
	global_item_size[1]=height % local_item_size[1] == 0 ? height : (height / local_item_size[1] + 1) * local_item_size[1];
	//printf("local=%d,%d global=%d,%d\n",(int)local_item_size[0],(int)local_item_size[1],(int)global_item_size[0],(int)global_item_size[1]);
	
	//run gradient
	ret = clEnqueueNDRangeKernel(command_queue, kernel_gradient, 2, NULL, global_item_size, local_item_size, 0, NULL, &event_gradient);
	clWaitForEvents(1, &event_gradient);
	
	/* //don't read diff and direct
	ret = clEnqueueReadBuffer(command_queue, diff_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), diff, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, direct_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), direct, 0, NULL, NULL);
	*/
	
	//run integral
	ret = clEnqueueNDRangeKernel(command_queue, kernel_integral, 2, NULL, global_item_size, local_item_size, 0, NULL, &event_integral);
	clWaitForEvents(1, &event_integral);
	ret = clEnqueueReadBuffer(command_queue, result_mem_obj, CL_TRUE, 0, DATA_SIZE * sizeof(float), result, 0, NULL, &event_readbuffer);
	clWaitForEvents(1, &event_readbuffer);
	
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
	int t_width,t_height;
	FILE *fp;
	if( (fp=fopen("default_radius","r")) == NULL){
		puts("No Default File!!");
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
	search_prime();
	printf("%s\n",argv[1]);
	char OrigFileName[255];
	char FileName[255];
	if(argc<2){
		puts("Please input a image!!");
		exit(1);
	}
	strcpy(OrigFileName,argv[1]);
	//load image on gray
	IplImage *img = cvLoadImage(OrigFileName,CV_LOAD_IMAGE_COLOR|CV_LOAD_IMAGE_ANYDEPTH);
	int channel=img->nChannels;
	int depth=img->depth;
	int width=img->width;
	int height=img->height;
	IplImage *img2; //= cvCreateImage(cvSize(width,height),depth,channel); 			//downsize
	IplImage *img3= cvCreateImage(cvSize(width,height),depth,channel);				//upsize
	IplImage *img_out2; //= cvCreateImage(cvSize(width,height),depth,channel); 		//downsize
	IplImage *img_out3= cvCreateImage(cvSize(width,height),IPL_DEPTH_16U,channel);			//upsize
	IplImage *ch[3];
	IplImage *ch_out[3];
	unsigned int value; //ptr for IPL_DEPTH_16U
	
	printf("Input Depth=%d\n",depth);
	
	for(int color=0;color<3;++color){
		ch[color]=cvCreateImage(cvSize(width,height),depth,1);
	}
	cvSplit(img, ch[0], ch[1], ch[2], NULL);
	for(int color=0;color<3;++color){
		sprintf(FileName,"%s_Color%d.png",OrigFileName,color);
		cvSaveImage(FileName,ch[color]);
		cvReleaseImage( &ch[color] );
	}
	
	//get Image data
	//unsigned char *img_data= (unsigned char*) malloc(sizeof(unsigned char)*width*height*channel*16);
	float *orig_data= (float*) malloc(sizeof(float)*width*height*4);
	float *GMEMD_diff= (float*) malloc(sizeof(float)*width*height*4);
	//float *GMEMD_diff_gauss= (float*) malloc(sizeof(float)*width*height);
	float *GMEMD_direct= (float*) malloc(sizeof(float)*width*height*4);
	float *imf_image= (float*) malloc(sizeof(float)*width*height*4);
	//float *imf_count= (float*) malloc(sizeof(float)*width*height);
	float tmp;

	if(orig_data==NULL) printf("orig_data malloc error!!\n");
	if(imf_image==NULL) printf("imf_image malloc error!!\n");
/*
	for(int i=0;i<img->width;++i){
		for(int j=0;j<img->height;++j){
			for(int k=0;k<img->nChannels;++k){
				img_data[(j*img->width+i)*img->nChannels+k]=img->imageData[j*img->widthStep+i*img->nChannels+k];
				orig_data[j*width+i]=(float)img_data[(j*img->width+i)*img->nChannels+k];
			}
		}
	}
*/

	//printf("size=%d*%d depth=%d channel=%d\n",width,height,depth,channel);
	/*
	sprintf(FileName,"%s_gray.png",OrigFileName);
	cvSaveImage(FileName,img);
	*/
	init_cl(width*height*4);
	
	//float now_data,target_data;
	//for(int p=0,r=1;p<P_size;r=prime[p++]){
	//for(int p=24,r=prime[p];p<P_size;r=prime[p++]){
	//for(int r=1;r<=64;r*=2){
	//for(int r=1;r<=25;++r){
	//for(int r=26;r<=30;++r){
	//for(int r=1;r<=30;++r){
	for(int i=0,r;i<default_radius_len;++i){
		start_time=clock();
		r=default_radius[i];
		ratio=default_ratio[i];
		t_width=(int)(width/ratio);
		t_height=(int)(height/ratio);
		
		build_list(r);
		//printf("ratio=%f\tradius=%d\tlen=%d\t ",ratio,r,ar_len);
		printf("effective_radius=%.1f\t(ratio=%.1f,radius=%d)\t ",(ratio*r),ratio,r);
		//system("pause");
		img2=cvCreateImage(cvSize(t_width,t_height),depth,channel);
		img_out2=cvCreateImage(cvSize(t_width,t_height),IPL_DEPTH_16U,channel);
		cvResize( img, img2, CV_INTER_LINEAR );
		for(int color=0;color<3;++color){
			ch[color]=cvCreateImage(cvSize(t_width,t_height),depth,1);
			ch_out[color]=cvCreateImage(cvSize(t_width,t_height),IPL_DEPTH_16U,1);
		}
		//system("pause");
		cvSplit(img2, ch[0], ch[1], ch[2], NULL);
		
		/*
	for(int i=0;i<img2->width;++i){
		for(int j=0;j<img2->height;++j){
			for(int k=0;k<img2->nChannels;++k){
				img_data[(j*img2->width+i)*img2->nChannels+k]=img2->imageData[j*img2->widthStep+i*img2->nChannels+k];
				orig_data[j*img2->width+i]=(float)img_data[(j*img2->width+i)*img2->nChannels+k];
			}  
		}
	}
	*/
		
		for(int color=0;color<3;++color){

			for(int i=0;i<t_width;++i){
				for(int j=0;j<t_height;++j){
					if(depth==8){
						orig_data[j*t_width+i]=(float)(unsigned char)ch[color]->imageData[j*ch[color]->widthStep+i];
					}
					else if(depth==16){
						//orig_data[j*t_width+i]=(float)(unsigned char)ch[color]->imageData[j*ch[color]->widthStep+i*2];
						//orig_data[j*t_width+i]+=(float)(unsigned char)ch[color]->imageData[j*ch[color]->widthStep+i*2+1]*256;
						orig_data[j*t_width+i]=(float) (     (int)(unsigned char)(ch[color]->imageData[j*ch[color]->widthStep+i*2])
													     + ( (int)(unsigned char)(ch[color]->imageData[j*ch[color]->widthStep+i*2+1]) ) * 256 );
					}
					//printf("%d %d %f\n",i,j,orig_data[j*t_width+i]);
				}
			}
			
			run_cl(orig_data,imf_image,GMEMD_diff,GMEMD_direct,t_width,t_height,r);
			/*
			for(int i=0;i<t_width;++i){
				for(int j=0;j<t_height;++j){
					imf_image[j*t_width+i]=-imf_image[j*t_width+i];
				}
			}
			*/
			normalize_img(imf_image,t_width,t_height,1.0);
			
			for(int j=0;j<t_height;++j){
				for(int i=0;i<t_width;++i){
					//ch_out[color]->imageData[j*ch_out[color]->widthStep+i]=(unsigned short)(imf_image[j*t_width+i]*65535);
					
					//cvSet2D(ch_out[color],j,i,(short unsigned int)(imf_image[j*t_width+i]*65535));
					
					//ptr=cvPtr2D(ch_out[color],j,i);
					//*ptr=(unsigned short)(imf_image[j*t_width+i]*65535);
					
					value=(unsigned int)(imf_image[j*t_width+i]*65535);
					ch_out[color]->imageData[j*ch_out[color]->widthStep+i*2+1]=(unsigned char)(int)(value/256);
					ch_out[color]->imageData[j*ch_out[color]->widthStep+i*2]=(unsigned char)(int)(value%255);
					//ch_out[color]->imageData[j*ch_out[color]->widthStep+i*2]=(unsigned short)(int)(value);
					
					//ptr=ch_out[color]->imageData[j*ch_out[color]->widthStep+i];
					//*ptr=(unsigned short)(imf_image[j*t_width+i]*255);
					

					
					//cvNamedWindow("test");
					//cvShowImage("test", ch_out[color]);
					//sprintf(FileName,"%s_GMEMD_CBIMF%d_C%d.png",OrigFileName,(int)(ratio*r),color);
					//printf("%s",FileName);
					//cvSaveImage(FileName,ch_out);
				}
			}
			
			/*
			sprintf(FileName,"%s_GMEMD_color_BIMF%d_ch[%d].png",OrigFileName,(int)(ratio*r),color);
			cvSaveImage(FileName,ch[color]);
			*/
		}
		
		cvMerge(ch_out[0], ch_out[1], ch_out[2], NULL, img_out2);
		
		cvResize( img_out2, img_out3, CV_INTER_LINEAR );
		
		//sprintf(FileName,"%s_GMEMD_ColorSpatialFrame%.2f(%.2f*%d).png",OrigFileName,(ratio*r),ratio,r);
		sprintf(FileName,"%s_GMEMD_ColorSpatialFrame%.1f(%.1fx%d).png",OrigFileName,(ratio*r),ratio,r);
		//printf("%s",FileName);
		cvSaveImage(FileName,img_out3);
		
		
		//puts("run cl...");
		//run_cl(orig_data,imf_image,GMEMD_diff,GMEMD_direct,t_width,t_height,r);
		
		//puts("	done!!");
		/*
		//reset imf_image
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				imf_image[j*width+i]=0.0;
			}
		}
		
		//calc gradient and direction
		lbp_grad(img_data,lbp_diff,lbp_direct,height,width);
		
		//gaussian diff
		img_gaussian(lbp_diff,lbp_diff_gauss,height,width);
		
		//intergral to imf
		integral_imf(imf_image,lbp_diff,lbp_direct,height,width);
		*/
		
		
		//output image
		
		/*
		normalize_img(GMEMD_diff,t_width,t_height,1.0);
		for(int j=0;j<t_height;++j){
			for(int i=0;i<t_width;++i){
				img2->imageData[j*img2->widthStep+i]=(char)(GMEMD_diff[j*t_width+i]*255);
			}
		}
		sprintf(FileName,"%s_GMEMD_diff%dx%d=%d.png",OrigFileName,ratio,r,ratio*r);
		cvResize( img2, img3, CV_INTER_LINEAR );
		*/
		
		//cvSaveImage(FileName,img3);
		

		/*
		img_gaussian(lbp_diff,lbp_diff_gauss,t_height,t_width);
		normalize_img(lbp_diff_gauss,t_width,t_height,1.0);
		for(int j=0;j<t_height;++j){
			for(int i=0;i<t_width;++i){
				img2->imageData[j*img2->widthStep+i]=(char)(lbp_diff_gauss[j*t_width+i]*255);
			}
		}
		sprintf(FileName,"image_lbpgrad_diff_gauss%d.png",r);
		cvSaveImage(FileName,img2);
		*/
		
		/* 
		normalize_img(GMEMD_direct,t_width,t_height,1.0);
		for(int j=0;j<t_height;++j){
			for(int i=0;i<t_width;++i){
				img2->imageData[j*img2->widthStep+i]=(unsigned char)(GMEMD_direct[j*t_width+i]*255);
			}
		}
		sprintf(FileName,"%s_GMEMD_direct%dx%d=%d.png",OrigFileName,ratio,r,ratio*r);
		cvResize( img2, img3, CV_INTER_LINEAR );
		cvSaveImage(FileName,img3);
		*/ 
		
		/* 
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				imf_image[j*width+i]=-imf_image[j*width+i];
			}
		}
		*/ 
		/* 
		normalize_img(imf_image,t_width,t_height,1.0);
		for(int j=0;j<t_height;++j){
			for(int i=0;i<t_width;++i){
				img2->imageData[j*img2->widthStep+i]=(unsigned char)(imf_image[j*t_width+i]*255);
			}
		}
		sprintf(FileName,"%s_GMEMD_imf%fx%d=%.2f.png",OrigFileName,ratio,r,ratio*r);
		cvResize( img2, img3, CV_INTER_LINEAR );
		cvSaveImage(FileName,img3);
		*/ 
		/*
		//normalize_img(lbp_diff_gauss,width,height,2.3026);
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				imf_image[j*width+i]*=lbp_diff_gauss[j*width+i];
			}
		}
		normalize_img(imf_image,width,height,1.0);
		for(int j=0;j<height;++j){
			for(int i=0;i<width;++i){
				img2->imageData[j*img2->widthStep+i]=(unsigned char)(imf_image[j*width+i]*255);
			}
		}
		sprintf(FileName,"image_lbpgrad_bimf%d.png",r);
		cvSaveImage(FileName,img2);
		*/
		
		cvReleaseImage( &img2 );
		cvReleaseImage( &img_out2 );
		
		for(int color=0;color<3;++color){
			cvReleaseImage( &ch[color] );
			cvReleaseImage( &ch_out[color] );
		}
		end_time=clock();
		used_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;
		printf("time=%f",used_time);
		puts("	done");
	}
	system("pause");
	return 0;
}


