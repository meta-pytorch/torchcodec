#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class DummyDeviceInterface : public DeviceInterface {
 public:
  DummyDeviceInterface(const torch::Device& device) : DeviceInterface(device) {}

  virtual ~DummyDeviceInterface() {}

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override {}

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override {}

 private:
  std::unique_ptr<FilterGraph> filterGraphContext_;
};

namespace {
static bool g_dummy = registerDeviceInterface(
    DeviceInterfaceKey(torch::kPrivateUse1),
    [](const torch::Device& device) {
      return new DummyDeviceInterface(device);
    });
} // namespace
} // namespace facebook::torchcodec
