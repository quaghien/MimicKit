# Chạy MimicKit trên Laptop GPU yếu (RTX 3050 Ti / 4GB VRAM)

Hướng dẫn này ghi lại các bước và thay đổi code cần thiết để chạy MimicKit với Isaac Lab trên laptop hybrid graphics có VRAM thấp.

## Cấu hình máy

- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop (4GB VRAM)
- **iGPU:** Intel UHD Graphics (TGL GT1)
- **Driver NVIDIA:** 535.x (xem phần bên dưới)
- **OS:** Ubuntu 24.04, Kernel 6.17
- **Isaac Sim:** 5.1.0

---

## 1. Driver NVIDIA

### Driver yêu cầu

Isaac Sim 5.1 yêu cầu driver NVIDIA **535.x trên Linux**. Driver mới hơn (ví dụ 595.x) gây crash ở plugin `librtx.scenedb.plugin.so`.

Driver được test và hoạt động: **535.288.01** (cài qua `ubuntu-drivers` hoặc `.run` file).

> **Lưu ý:** Driver 535.171 không build được DKMS trên kernel 6.17 (`os-interface.h` không tìm thấy). Dùng 535.288 thay thế.

### Cách cài driver 535

```bash
sudo apt install nvidia-driver-535
# hoặc nếu cần version cụ thể:
sudo ubuntu-drivers install nvidia:535
```

Nếu máy bật **Secure Boot**, cần enroll MOK key sau khi cài:
```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
# Reboot → chọn "Enroll MOK" → nhập password → reboot lại
```

### Bắt buộc dùng chế độ NVIDIA (không dùng on-demand)

```bash
sudo prime-select nvidia
sudo reboot
```

Kiểm tra:
```bash
nvidia-smi        # phải thấy GPU NVIDIA
glxinfo | grep "OpenGL renderer"  # phải thấy "GeForce RTX..."
```

---

## 2. Sửa driver-requirements.json

Isaac Sim đọc driver version qua Vulkan API. Trên một số hệ thống, Vulkan báo version khác nvidia-smi (ví dụ `535.32.01` thay vì `535.288.01`). Version thấp hơn sẽ bị block khiến viewport đen.

Sửa file:
```
/home/<user>/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/driver-requirements.json
```

Tìm block `linux-x86_64` trong section `rtx`, đổi `"minor": 129` thành `"minor": 0`:

```json
{
    "message": "The minimum Omniverse RTX requirement on Linux",
    "blocked": true,
    "max": {
        "major": 535,
        "minor": 0,
        "patch": 0,
        "build": 0
    }
}
```

---

## 3. Thay đổi code trong `mimickit/engines/isaac_lab_engine.py`

### 3a. Thêm `kit_args` vào AppLauncher

Trong hàm `_create_simulator()`, thêm `kit_args` để giảm VRAM RTX renderer khi visualize và tắt driver version check:

```python
self._app_launcher = AppLauncher({
    "headless": not visualize,
    "device": self._device,
    "enable_cameras": record_video or visualize,
    "kit_args": (
        "--/rtx/verifyDriverVersion=false "
        "--/rtx/post/dlss/execMode=0 "
        "--/rtx/post/aa/op=0 "
        "--/rtx/shadows/enabled=false "
        "--/rtx/post/histogram/enabled=false "
        "--/renderer/resolution/width=1280 "
        "--/renderer/resolution/height=720 "
        "--/rtx/resourcemanager/maxTextureMemoryMB=256 "
        "--/rtx/materialDb/enabled=false "
        "--/rtx/sceneDb/maxInstances=65536"
    ),
})
```

| Flag | Lý do |
|------|-------|
| `verifyDriverVersion=false` | Tắt check driver version trong Kit (không đủ khi chỉ sửa json) |
| `maxTextureMemoryMB=256` | Giới hạn texture cache RTX ở 256MB thay vì ~1.5GB |
| `shadows/enabled=false` | Tắt bóng đổ để tiết kiệm VRAM |
| `materialDb/enabled=false` | Tắt preload material DB (~100-200MB) |
| `aa/op=0`, `dlss/execMode=0` | Tắt AA và DLSS |

### 3b. Giảm PhysX GPU buffer khi visualize

RTX renderer chiếm ~2GB VRAM, PhysX cần phần còn lại. Thêm điều kiện `if visualize:` để chỉ giảm buffer khi có GUI:

```python
sim_cfg.physx.bounce_threshold_velocity = 0.2
sim_cfg.physx.max_position_iteration_count = 4
sim_cfg.physx.max_velocity_iteration_count = 0
if visualize:
    # RTX renderer takes ~2GB; shrink PhysX buffers to fit remaining VRAM
    sim_cfg.physx.gpu_max_rigid_contact_count = 64 * 1024
    sim_cfg.physx.gpu_max_rigid_patch_count = 32 * 1024
    sim_cfg.physx.gpu_heap_capacity = 64 * 1024 * 1024
    sim_cfg.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024
    sim_cfg.physx.gpu_max_num_partitions = 4
```

Khi `--visualize false` (train headless), không set → dùng default Isaac Lab.

---

## 4. Lệnh chạy

### Test với visualization (xem humanoid)

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --num_envs 1 \
  --visualize true \
  --mode test \
  --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```

> Dùng `--num_envs 1` khi visualize vì RTX chiếm ~2GB VRAM, không còn đủ cho nhiều env.

### Train headless (khuyến nghị cho 4GB VRAM)

```bash
# Thử 2048 envs trước (an toàn hơn với 4GB)
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --num_envs 2048 \
  --visualize false \
  --mode train \
  --out_dir output/

# Nếu CUDA OOM, giảm xuống 1024
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --num_envs 1024 \
  --visualize false \
  --mode train \
  --out_dir output/
```

> **Lưu ý VRAM:** 4096 envs (default) yêu cầu ~16-40GB VRAM như các máy A100/V100. Với 4GB, dùng 1024-2048 envs headless.

---

## Tóm tắt thay đổi

| Thay đổi | File | Lý do |
|----------|------|-------|
| Cài driver 535.288 | hệ thống | Driver 595+ crash Isaac Sim 5.1 RTX plugin |
| `prime-select nvidia` | hệ thống | Hybrid on-demand mode dùng Intel GPU, Isaac Sim không nhận NVIDIA |
| Enroll MOK key | hệ thống | Secure Boot chặn kernel module NVIDIA |
| Sửa `driver-requirements.json` | isaaclab package | Vulkan báo version thấp hơn thực tế, bị block |
| Thêm `kit_args` vào AppLauncher | `isaac_lab_engine.py` | Giảm VRAM RTX, tắt driver check trong Kit |
| Giảm PhysX buffer khi visualize | `isaac_lab_engine.py` | RTX + PhysX không vừa 4GB nếu buffer mặc định |
