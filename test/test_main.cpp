//
// Created by zhou on 2026/4/8.
//

#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("DIOYF");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test ... \n";

    return RUN_ALL_TESTS();
}