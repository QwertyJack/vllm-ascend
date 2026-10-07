// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
// Pinned Mooncake 0.3.11.post1 calls the legacy ADXL API, whose rank-table
// generator does not support A5. Forward that public API to the same CANN
// library's native HIXL API. Its factory selects hixl_cs for A5, with actual
// host-provisioned endpoints. No rank-table/address invention or host staging.
// Opt-in LD_PRELOAD artifact: no system/Mooncake binaries are overwritten.
#include <adxl/adxl_engine.h>
#include <acl/acl.h>
#include <hixl/hixl.h>

#include <cstdio>

// Exported by the pinned libcann_hixl.so, but not installed as a public header.
// Generates the resource JSON from the real local physical NPU topology.
namespace hixl {
Status TransLocalCommRes(int32_t device, AscendString& resource);
}

namespace adxl {
static_assert(static_cast<int>(MEM_DEVICE) == static_cast<int>(hixl::MEM_DEVICE));
static_assert(static_cast<int>(MEM_HOST) == static_cast<int>(hixl::MEM_HOST));
static_assert(static_cast<int>(READ) == static_cast<int>(hixl::READ));
static_assert(static_cast<int>(WRITE) == static_cast<int>(hixl::WRITE));
static_assert(static_cast<int>(TransferStatus::WAITING) == static_cast<int>(hixl::TransferStatus::WAITING));
static_assert(static_cast<int>(TransferStatus::COMPLETED) == static_cast<int>(hixl::TransferStatus::COMPLETED));
static_assert(static_cast<int>(TransferStatus::TIMEOUT) == static_cast<int>(hixl::TransferStatus::TIMEOUT));
static_assert(static_cast<int>(TransferStatus::FAILED) == static_cast<int>(hixl::TransferStatus::FAILED));
class AdxlEngine::AdxlEngineImpl {
 public:
  hixl::Hixl native;
};

AdxlEngine::AdxlEngine() : impl_(std::make_unique<AdxlEngineImpl>()) {}
AdxlEngine::~AdxlEngine() = default;

Status AdxlEngine::Initialize(const AscendString& local_engine, const std::map<AscendString, AscendString>& options) {
  std::fprintf(stderr, "Dots3 Mooncake: forwarding legacy ADXL to native HIXL on %s\n", local_engine.GetString());
  auto native_options = options;
  for (const auto& entry : options) {
    std::fprintf(stderr, "Dots3 Mooncake option: %s=%s\n", entry.first.GetString(), entry.second.GetString());
  }
  // Canonical HIXL key, preserving Mooncake's disabled-buffer-pool setting.
  native_options[hixl::OPTION_BUFFER_POOL] = "0:0";
  // The pinned July factory selects native HIXL when supplied its generated
  // 1.3 LocalCommRes. Use CANN's own topology generator, not a hand-written EID.
  int32_t logical_device = -1;
  int32_t physical_device = -1;
  if (aclrtGetDevice(&logical_device) != ACL_SUCCESS ||
      aclrtGetPhyDevIdByLogicDevId(logical_device, &physical_device) != ACL_SUCCESS) {
    return hixl::FAILED;
  }
  hixl::AscendString resource;
  const auto generated = hixl::TransLocalCommRes(physical_device, resource);
  if (generated != hixl::SUCCESS) return generated;
  native_options[hixl::OPTION_LOCAL_COMM_RES] = resource;
  std::fprintf(stderr, "Dots3 native HIXL physical device %d resources: %s\n", physical_device, resource.GetString());
  return impl_->native.Initialize(local_engine, native_options);
}

void AdxlEngine::Finalize() { impl_->native.Finalize(); }

Status AdxlEngine::RegisterMem(const MemDesc& mem, MemType type, MemHandle& handle) {
  const hixl::MemDesc native_mem{mem.addr, mem.len};
  return impl_->native.RegisterMem(native_mem, static_cast<hixl::MemType>(type), handle);
}

Status AdxlEngine::DeregisterMem(MemHandle handle) { return impl_->native.DeregisterMem(handle); }

Status AdxlEngine::Connect(const AscendString& peer, int32_t timeout) { return impl_->native.Connect(peer, timeout); }

Status AdxlEngine::Disconnect(const AscendString& peer, int32_t timeout) {
  return impl_->native.Disconnect(peer, timeout);
}

Status AdxlEngine::TransferSync(const AscendString& peer, TransferOp operation,
                                const std::vector<TransferOpDesc>& descriptors, int32_t timeout) {
  std::vector<hixl::TransferOpDesc> native_descriptors;
  native_descriptors.reserve(descriptors.size());
  for (const auto& entry : descriptors) {
    native_descriptors.push_back({entry.local_addr, entry.remote_addr, entry.len});
  }
  return impl_->native.TransferSync(peer, static_cast<hixl::TransferOp>(operation), native_descriptors, timeout);
}

Status AdxlEngine::TransferAsync(const AscendString& peer, TransferOp operation,
                                 const std::vector<TransferOpDesc>& descriptors, const TransferArgs& args,
                                 TransferReq& request) {
  (void)args;  // Legacy TransferArgs contains reserved bytes only.
  std::vector<hixl::TransferOpDesc> native_descriptors;
  native_descriptors.reserve(descriptors.size());
  for (const auto& entry : descriptors) {
    native_descriptors.push_back({entry.local_addr, entry.remote_addr, entry.len});
  }
  const hixl::TransferArgs native_args{};
  return impl_->native.TransferAsync(peer, static_cast<hixl::TransferOp>(operation), native_descriptors, native_args,
                                     request);
}

Status AdxlEngine::GetTransferStatus(const TransferReq& request, TransferStatus& status) {
  hixl::TransferStatus native_status;
  const auto result = impl_->native.GetTransferStatus(request, native_status);
  if (result == hixl::SUCCESS) status = static_cast<TransferStatus>(native_status);
  return result;
}

Status AdxlEngine::SendNotify(const AscendString& peer, const NotifyDesc& notify, int32_t timeout) {
  return impl_->native.SendNotify(peer, {notify.name, notify.notify_msg}, timeout);
}

Status AdxlEngine::GetNotifies(std::vector<NotifyDesc>& notifies) {
  std::vector<hixl::NotifyDesc> native_notifies;
  const auto result = impl_->native.GetNotifies(native_notifies);
  for (const auto& entry : native_notifies) {
    notifies.push_back({entry.name, entry.notify_msg});
  }
  return result;
}

Status AdxlEngine::GetCapability(FeatureType feature, int32_t& value) {
  return hixl::Hixl::GetCapability(static_cast<hixl::FeatureType>(feature), value);
}
}  // namespace adxl
