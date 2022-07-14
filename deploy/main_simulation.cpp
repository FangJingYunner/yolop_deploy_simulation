#include "yolov5.hpp"
//#include "zedcam.hpp"
#include <csignal>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <customized_msgs/BboxList.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>

static volatile bool keep_running = true;

using namespace std;
using namespace nvinfer1;
using namespace cv;
using namespace sensor_msgs;

void keyboard_handler(int sig) {
    // handle keyboard interrupt
    if (sig == SIGINT)
        keep_running = false;
}

image_transport::Publisher pub_seg_results;
ros::Publisher pub_det_results;
//ros::Publisher pub_seg_results;
ros::Subscriber sub_img;

void* buffers[4];

#define NMS_THRESH 0.1
#define CONF_THRESH 0.85
#define BATCH_SIZE 1

int inputIndex;
int output_det_index;
int output_seg_index;
int output_lane_index;

//float* det_out;
//int* seg_out;
//int* lane_out;

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
cudaStream_t stream;

const char* IMG_TOPIC = "/ipm_raw";
const char* PUB_TOPIC_DET = "/yolop_det/results";
const char* PUB_TOPIC_SEG = "/yolop_seg/results";

//image_transport::ImageTransport it;

static float det_out[BATCH_SIZE * OUTPUT_SIZE];
static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
static int lane_out[BATCH_SIZE * IMG_H * IMG_W];

void img_callback(const sensor_msgs::ImageConstPtr &img_sub) {

    customized_msgs::BboxList bboxlist;
    bboxlist.header = img_sub->header;

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_sub, "rgb8");
//    cout << "stamp : " << img_sub->header.stamp << endl;
    cv::Mat img = cv_ptr->image;

//    cout<<1<<endl;
//                imshow("RawImg",img);
//        waitKey(6000);

//    img_callback_main(img, true);
//        clock_t starttime = clock();
//        cout<<1<<endl;
    cv::cuda::GpuMat gpu_img(INPUT_H, INPUT_W, CV_8UC3);
    gpu_img.upload(img);

//    cout<<INPUT_W<<endl;
//    cout<<INPUT_H<<endl;

    preprocess_img_gpu(gpu_img, (float*)buffers[inputIndex], INPUT_W, INPUT_H);

    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, seg_out);
    // sotore lane results
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_out);
    cv::Mat seg_res(INPUT_H, INPUT_W, CV_32S);
    cv::Mat lane_res(INPUT_H, INPUT_W, CV_32S);



    auto start = std::chrono::system_clock::now();
    // cuCtxPushCurrent(ctx);
    doInference(*context, stream, buffers, det_out, seg_out, lane_out, BATCH_SIZE);
    // cuCtxPopCurrent(&ctx);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    char key = ' ';
    std::vector<Yolo::Detection> batch_res;

    nms(bboxlist, det_out, CONF_THRESH, NMS_THRESH);

    pub_det_results.publish(bboxlist);


//    for(int i = 0;i < batch_res.size();i++){
//        cout<<batch_res[i].bbox[0]<<" "<<batch_res[i].bbox[1]<<" "<<batch_res[i].bbox[2]<<" "<<batch_res[i].bbox[3]<<endl;
//        cout<<batch_res[i].class_id<<endl;
//        cout<<batch_res[i].conf<<endl;
//    }
    cv::resize(tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
    cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);


    visualization(gpu_img, seg_res, lane_res, batch_res, key);

    seg_res.convertTo(seg_res, CV_8U,50);

//    for (int row = 0; row < seg_res.rows; ++row) {
////        uchar* pdata = cvt_img_cpu.data + row * cvt_img_cpu.step;
//        for (int col = 0; col < seg_res.cols; ++col) {
////            seg_res.at<cv::Vec3b>(row, col) = seg_res.at<cv::Vec3b>(row, col) * 40;
//            if(seg_res.at<int>(row, col) > 0)
//                cout<<seg_res.at<int>(row, col)<<endl;
//        }
//    }
    sensor_msgs::ImagePtr seg_msg =cv_bridge::CvImage(img_sub->header, "mono8", seg_res).toImageMsg();
    pub_seg_results.publish(seg_msg);


}
int main(int argc, char** argv) {



    ros::init(argc,argv,"yolox");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    sub_img = nh.subscribe(IMG_TOPIC, 1, &img_callback);
    pub_det_results = nh.advertise<customized_msgs::BboxList>(PUB_TOPIC_DET, 1);
//    pub_seg_results = nh.advertise<customized_msgs::BboxList>(PUB_TOPIC_SEG, 1);
    pub_seg_results = it.advertise("/wood_ros/front_fisheye", 1);

    signal(SIGINT, keyboard_handler);
    cudaSetDevice(DEVICE);
    // CUcontext ctx;
    // CUdevice device;
    // cuInit(0);
    // cuDeviceGet(&device, 0);
    // cuCtxCreate(&ctx, 0, device);

    std::string wts_name = "/home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/yolop_simulation.wts";
    std::string engine_name = "/home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/yolop_simulation_fp16_5class.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        // std::cerr << "read " << engine_name << " error!" << std::endl;
        std::cout << "Building engine..." << std::endl;
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "Engine has been built and saved to file." << std::endl;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare data ---------------------------
    static float det_out[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);
//    void* buffers[4];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    output_det_index = engine->getBindingIndex(OUTPUT_DET_NAME);
    output_seg_index = engine->getBindingIndex(OUTPUT_SEG_NAME);
    output_lane_index = engine->getBindingIndex(OUTPUT_LANE_NAME);
    assert(inputIndex == 0);
    assert(output_det_index == 1);
    assert(output_seg_index == 2);
    assert(output_lane_index == 3);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_det_index], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_seg_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers[output_lane_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    // Create stream
//    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

//    // create zed
//    auto zed = create_camera();
//    sl::Resolution image_size = zed->getCameraInformation().camera_configuration.resolution;
//    sl::Mat img_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
//    cv::cuda::GpuMat img_ocv = slMat2cvMatGPU(img_zed);
//    cv::cuda::GpuMat cvt_img(image_size.height, image_size.width, CV_8UC3);
//
//    // store seg results
//    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, seg_out);
//    // sotore lane results
//    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_out);
//    cv::Mat seg_res(image_size.height, image_size.width, CV_32S);
//    cv::Mat lane_res(image_size.height, image_size.width, CV_32S);
//
//    char key = ' ';
//    while (keep_running and key != 'q') {
//        // retrieve img
//        if (zed->grab() != sl::ERROR_CODE::SUCCESS) continue;
//        zed->retrieveImage(img_zed, sl::VIEW::LEFT, sl::MEM::GPU);
//        cudaSetDevice(DEVICE);
//        cv::cuda::cvtColor(img_ocv, cvt_img, cv::COLOR_BGRA2BGR);
//
//        // preprocess ~3ms
//        preprocess_img_gpu(cvt_img, (float*)buffers[inputIndex], INPUT_W, INPUT_H); // letterbox
//
//        // buffers[inputIndex] = pr_img.data;
//        // Run inference
//        auto start = std::chrono::system_clock::now();
//        // cuCtxPushCurrent(ctx);
//        doInference(*context, stream, buffers, det_out, seg_out, lane_out, BATCH_SIZE);
//        // cuCtxPopCurrent(&ctx);
//        auto end = std::chrono::system_clock::now();
//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//
//        // postprocess ~0ms
//        std::vector<Yolo::Detection> batch_res;
//        nms(batch_res, det_out, CONF_THRESH, NMS_THRESH);
//        cv::resize(tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
//        cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);
//
//        // show results
//        //std::cout << res.size() << std::endl;
//        visualization(cvt_img, seg_res, lane_res, batch_res, key);
//    }
//    // destroy windows

    ros::spin();

#ifdef SHOW_IMG
    cv::destroyAllWindows();
#endif

//    // close camera
//    img_zed.free();
//    zed->close();
//    delete zed;
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[output_det_index]));
    CUDA_CHECK(cudaFree(buffers[output_seg_index]));
    CUDA_CHECK(cudaFree(buffers[output_lane_index]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
