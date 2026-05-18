# Giai đoạn 2

## Mục tiêu

Giai đoạn này nhằm hiểu:

- lệnh `python mimickit/run.py ...` đi qua những hàm nào
- `run.py` chỉ điều phối gì, không tự chứa logic imitation ở đâu
- env, agent, engine được dựng theo thứ tự nào
- train 1 GPU và train nhiều GPU khác nhau thế nào trong repo này

---

## Bước 4. Đọc `mimickit/run.py`

File chính:

- `mimickit/run.py`

Đây là entry point của repo. Khi chạy train/test, hầu như mọi thứ đều bắt đầu từ đây.

## Call graph ngắn nhất cần nhớ

```txt
main(argv)
  -> load_args(argv)
  -> chọn devices
  -> spawn process nếu có nhiều device
  -> run(rank, num_procs, device, master_port, args)
       -> mp_util.init(...)
       -> set_rand_seed(...)
       -> build_env(args, ...)
            -> env_builder.build_env(...)
       -> build_agent(args, env, ...)
            -> agent_builder.build_agent(...)
       -> nếu có model_file thì agent.load(...)
       -> train(...) hoặc test(...)
```

## 1. `main(argv)` làm gì

Trong [`mimickit/run.py`](/home/wanhin/hienhq/MimicKit/mimickit/run.py):

- gọi `load_args(argv)` để nạp toàn bộ tham số
- đọc `master_port`
- đọc `devices`, mặc định là `["cuda:0"]`
- nếu có nhiều device thì spawn thêm process bằng `torch.multiprocessing`
- process rank 0 chạy trên device đầu tiên, các rank còn lại chạy trên device tương ứng

### Ý chính

- `main` không train trực tiếp
- `main` chỉ chuẩn bị args, device, multiprocessing rồi gọi `run(...)`

---

## 2. `load_args(argv)` làm gì

Hàm này dùng [`mimickit/util/arg_parser.py`](/home/wanhin/hienhq/MimicKit/mimickit/util/arg_parser.py).

Flow:

1. nạp args từ command line trước
2. nếu có `--arg_file` thì đọc thêm file preset

### Điểm rất quan trọng

Ở implementation hiện tại, command line được load trước và `arg_file` chỉ điền thêm các key còn thiếu.

Nghĩa là:

- CLI có ưu tiên cao hơn `arg_file`
- `arg_file` đóng vai trò preset
- bạn có thể dùng preset rồi override bằng CLI

### Ví dụ

Giả sử preset có:

```txt
--engine_config data/engines/isaac_gym_engine.yaml
--num_envs 4096
```

Nếu chạy:

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --num_envs 512
```

thì giá trị được dùng thật sẽ là:

- `engine_config = isaac_lab_engine.yaml`
- `num_envs = 512`

chứ không phải giá trị trong preset.

---

## 3. `run(rank, num_procs, device, master_port, args)` làm gì

Đây là hàm điều phối cốt lõi.

Thứ tự chạy:

1. parse các tham số chính như `mode`, `num_envs`, `visualize`, `logger`, `model_file`, `out_dir`
2. gọi `mp_util.init(...)`
3. set random seed
4. set numpy formatting
5. tạo output dir
6. build env
7. build agent
8. nếu có checkpoint thì load model
9. nếu `mode == train` thì save config rồi train
10. nếu `mode == test` thì chạy test

### Ý nghĩa

- `run` là “bộ điều phối”
- `run` không định nghĩa reward, observation, action, motion imitation
- toàn bộ logic chuyên môn nằm sâu hơn trong `env`, `agent`, `engine`

### Câu trả lời nhanh cần nhớ

- đổi `--mode test` thì nhánh rẽ xảy ra trong `run(...)`
- `env` được tạo trước `agent`
- `agent` nhận `env` ngay lúc được build
- config được copy vào `out_dir` trong `save_config_files(...)`

---

## 4. `build_env(args, ...)` làm gì

Trong [`mimickit/run.py`](/home/wanhin/hienhq/MimicKit/mimickit/run.py), hàm này:

- đọc `env_config`
- đọc `engine_config`
- đọc thêm `video`
- gọi [`mimickit/envs/env_builder.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/env_builder.py)

Thực tế:

```txt
args
  -> env_config yaml
  -> engine_config yaml
  -> env_builder.build_env(...)
       -> chọn class env theo env_name
       -> env đó tự dùng engine_builder để dựng engine phù hợp
```

### Điều quan trọng trong `env_builder`

- `env_name` quyết định class env nào được tạo
- ví dụ `env_name: "deepmimic"` thì tạo `DeepMimicEnv`
- `env_builder` load cả `env_config` lẫn `engine_config`
- nếu trong `env_config` có block `engine`, nó có thể override một phần `engine_config`

### Ý nghĩa kiến trúc

- env là lớp logic bài toán
- engine là lớp simulator backend
- env builder là nơi nối hai lớp đó lại với nhau

---

## 5. `build_agent(args, env, device)` làm gì

Trong [`mimickit/learning/agent_builder.py`](/home/wanhin/hienhq/MimicKit/mimickit/learning/agent_builder.py):

- đọc `agent_config`
- lấy `agent_name`
- chọn class agent tương ứng
- ví dụ `agent_name: "PPO"` thì tạo `PPOAgent(config=agent_config, env=env, device=device)`
- sau khi tạo xong thì in tổng số parameter

### Điều phải note

- `agent` được tạo sau `env`
- `agent` nhận `env` trong constructor
- nghĩa là agent có thể hỏi env về `obs_space`, `action_space`, số env song song, v.v.

---

## 6. Sơ đồ quan hệ object

Sơ đồ đúng để nhớ:

```txt
args
  -> run.py
      -> env_builder
          -> DeepMimicEnv
              -> engine_builder
                  -> IsaacGymEngine / IsaacLabEngine / NewtonEngine
      -> agent_builder
          -> PPOAgent
              -> PPOModel
              -> MPOptimizer
              -> ExperienceBuffer
```

Hoặc nhìn theo hướng runtime:

```txt
agent <-> env <-> engine <-> simulator
```

Ý nghĩa:

- agent quyết định action
- env định nghĩa bài toán RL
- engine là adapter tới simulator
- simulator mới là nơi physics thực sự chạy

---

## 7. Train flow thật từ `run.py` tới PPO

Khi `mode == train`, flow lớn là:

1. `run.py` gọi `agent.train_model(...)`
2. `BaseAgent.train_model()` vào vòng lặp train
3. mỗi iteration gọi `_train_iter()`
4. `_train_iter()` gọi `_rollout_train(self._steps_per_iter)` để lấy dữ liệu
5. sau rollout xong mới gọi `_build_train_data()`
6. rồi gọi `_update_model()`
7. lặp tiếp iteration mới

### Cách hiểu ngắn

- pha 1: chạy env để lấy data
- pha 2: dùng data đó để học model
- không phải vừa lấy từng step xong là update ngay từng step đó

---

## 8. Bên trong PPO iteration diễn ra thế nào

Dựa trên [`mimickit/learning/base_agent.py`](/home/wanhin/hienhq/MimicKit/mimickit/learning/base_agent.py) và [`mimickit/learning/ppo_agent.py`](/home/wanhin/hienhq/MimicKit/mimickit/learning/ppo_agent.py):

### Pha rollout

- agent đang có `self._curr_obs`
- mỗi step:
  - actor chọn action
  - env step
  - buffer record `obs`, `action`, `reward`, `done`, `next_obs`, `a_logp`, ...
- lặp đủ `steps_per_iter`

### Pha build train data

- critic đánh giá `next_obs`
- tính `TD-lambda return`
- tính `advantage`
- chuẩn hóa advantage
- lưu `tar_val` và `adv` vào experience buffer

### Pha update

- critic update trước
- actor update sau
- mỗi bên lấy sample mini-batch từ experience buffer nhiều lần

### Điểm rất quan trọng

- trong code này, `actor_batch_size` và `critic_batch_size` được nhân với `num_envs`
- sau đó mới tính số mini-batch thực sự

Ví dụ:

- nếu `num_envs = 4096`
- `actor_batch_size = 4`

thì batch size thật cho actor sẽ là:

```txt
4 x 4096 = 16384 samples
```

---

## 9. Giai đoạn 2 cần tự trả lời được gì

Sau khi đọc xong giai đoạn này, bạn nên tự trả lời được:

1. `main()` chỉ điều phối gì
2. `run()` đổi flow ở đâu khi chuyển train sang test
3. env được build trước hay agent được build trước
4. agent có cầm `env` không
5. builder nào chọn `DeepMimicEnv`
6. builder nào chọn `PPOAgent`
7. builder nào chọn `IsaacLabEngine`
8. rollout và update tách nhau ở đâu

---

## 10. Train nhiều GPU khác gì 1 GPU

Đây là chỗ dễ hiểu sai nhất.

## Trường hợp 1 GPU

Ví dụ:

```bash
python mimickit/run.py --devices cuda:0
```

Khi đó:

- chỉ có 1 process
- chỉ có 1 env stack trên 1 device
- chỉ có 1 agent/model
- rollout được thu trên GPU đó
- gradient được tính và update ngay trên GPU đó

Nói ngắn:

- 1 GPU = 1 process train

---

## Trường hợp nhiều GPU

Ví dụ:

```bash
python mimickit/run.py --devices cuda:0 cuda:1
```

Khi đó `run.py` sẽ:

- spawn 2 process
- rank 0 chạy trên `cuda:0`
- rank 1 chạy trên `cuda:1`
- mỗi process tự build env riêng
- mỗi process tự build agent/model riêng

### Nhưng đây không phải 2 lần train độc lập

Điểm mấu chốt nằm ở:

- [`mimickit/util/mp_util.py`](/home/wanhin/hienhq/MimicKit/mimickit/util/mp_util.py)
- [`mimickit/learning/mp_optimizer.py`](/home/wanhin/hienhq/MimicKit/mimickit/learning/mp_optimizer.py)

Repo này dùng `torch.distributed` để:

- broadcast tham số ban đầu từ root process sang các process khác
- all-reduce gradient giữa các process mỗi lần `optimizer.step(loss)`

Nghĩa là:

- mỗi GPU có bản copy model riêng
- mỗi GPU rollout dữ liệu riêng trên env riêng
- mỗi GPU tự tính loss và gradient từ data cục bộ của nó
- rồi gradient của các GPU được gom trung bình lại
- sau đó tất cả GPU cùng update model theo cùng gradient trung bình đó

### Kết luận rất ngắn

- nhiều GPU trong repo này là **distributed data parallel kiểu tự viết đơn giản**
- không phải “mỗi GPU train một model khác nhau”

---

## 11. Vậy nhiều GPU có phải “gom data rồi train chung” không

Câu trả lời là: **gần đúng, nhưng chính xác hơn là gom gradient chứ không trực tiếp ghép hết data về một chỗ**.

### Cách nhìn đúng

Mỗi GPU làm:

1. tự rollout dữ liệu của mình
2. tự sample mini-batch từ buffer cục bộ của mình
3. tự tính gradient cục bộ
4. gọi `all_reduce(mean)` để trung bình gradient với GPU khác
5. update model đồng bộ

Nên về mặt ý tưởng:

- giống như data được chia ra nhiều GPU để tính gradient song song
- rồi gradient được gộp lại thành một bước update chung

### Không phải kiểu này

- GPU 0 train model A
- GPU 1 train model B
- train xong cuối cùng mới chọn một model

Repo này không làm kiểu đó.

---

## 12. Ví dụ cụ thể 1 GPU vs 2 GPU

Giả sử:

- `num_envs = 4096`
- `steps_per_iter = 32`
- chạy với 2 GPU: `cuda:0 cuda:1`

### Nếu là 1 GPU

- 1 process rollout `4096 env x 32 step`
- process đó tự train trên dữ liệu của nó

### Nếu là 2 GPU

- process 0 rollout `4096 env x 32 step` trên `cuda:0`
- process 1 rollout `4096 env x 32 step` trên `cuda:1`
- mỗi process có buffer riêng
- mỗi process tính gradient từ buffer riêng
- gradient hai bên được average lại
- cả hai model update đồng bộ nên vẫn giữ cùng trọng số

### Ý nghĩa thực tế

- throughput rollout tăng vì có thêm process/device
- tổng lượng dữ liệu dùng để ước lượng gradient trong cùng một iteration thực tế lớn hơn
- nhưng code không gom mọi sample về root process thành một mega-buffer duy nhất

---

## 13. Timeline logic của multi-GPU

Ví dụ với 2 GPU:

### Iteration k

- GPU 0 rollout data cục bộ
- GPU 1 rollout data cục bộ
- GPU 0 tính loss cục bộ
- GPU 1 tính loss cục bộ
- hai bên `all_reduce` gradient
- hai bên cùng update model

### Iteration k + 1

- cả hai bắt đầu rollout tiếp với model mới đã đồng bộ

### Cách nhớ ngắn

- rollout: riêng từng GPU
- model weights: luôn được đồng bộ
- update: dùng gradient trung bình từ tất cả GPU

---

## 14. Hai tầng song song hóa trong repo

Repo này có 2 tầng song song:

### Tầng 1. Nhiều env trong một process

- ví dụ `--num_envs 4096`
- một process điều khiển nhiều env song song trong cùng device/backend

### Tầng 2. Nhiều process trên nhiều device

- ví dụ `--devices cuda:0 cuda:1`
- mỗi process có env riêng, agent riêng, buffer riêng
- gradient được đồng bộ qua distributed communication

### Nhìn ngắn gọn

- `num_envs`: song song bên trong process
- `devices`: song song giữa các process

---

## 15. Tóm lại giai đoạn 2 bằng 5 câu

- `run.py` là entry point điều phối toàn bộ train/test nhưng không chứa logic imitation cụ thể.
- `load_args` nạp CLI trước, rồi `arg_file` chỉ bổ sung các key còn thiếu nên CLI có thể override preset.
- `env` được build trước, `agent` được build sau và agent nhận `env` trong constructor.
- Mỗi iteration của PPO gồm hai pha tách biệt: rollout dữ liệu trước, update model sau.
- Train nhiều GPU trong repo này là nhiều process cùng rollout và cùng tính gradient cục bộ, rồi gradient được đồng bộ trung bình để mọi GPU vẫn train chung một model logic.
