# Giai đoạn 1

## Bước 1. Đọc `README.md`

### Trả lời nhanh

- `MimicKit` là framework reinforcement learning cho motion imitation và motion control, được thiết kế gọn, ít phụ thuộc và hỗ trợ nhiều simulator backend.
- Các phương pháp imitation đang được hỗ trợ gồm: `DeepMimic`, `AMP`, `AWR`, `ASE`, `LCP`, `ADD`, `SMP`.
- Các engine đang được hỗ trợ gồm: `Isaac Gym`, `Isaac Lab`, `Newton`.
- Cách train/test tổng quát: chạy `python mimickit/run.py`, chọn mode bằng `--mode train|test`, chọn engine bằng `--engine_config`, chọn environment bằng `--env_config`, chọn agent bằng `--agent_config` hoặc dùng `--arg_file`; khi test thì thêm `--model_file`.
- Motion data nằm trong `data/motions/`, có thể trỏ tới một clip `.pkl` hoặc một dataset `.yaml` trong `data/datasets/`.
- Biểu diễn motion: mỗi frame chứa `root position (3D)`, `root rotation (3D exponential map)` và `joint rotations`; khớp 3D dùng exponential map, khớp 1D dùng góc quay 1 chiều.

### Ý quan trọng cần nhớ

- Repo hỗ trợ nhiều engine: `Isaac Gym`, `Isaac Lab`, `Newton`.
- Chọn engine bằng `--engine_config`.
- Chọn environment bằng `--env_config`.
- Chọn agent bằng `--agent_config`.
- Motion có thể là clip `.pkl` hoặc dataset `.yaml`.

### Viết lại bằng 3 câu

- `MimicKit` là một framework RL cho bài toán motion imitation/control với nhiều thuật toán và nhiều simulator backend dùng chung một kiến trúc.
- Pipeline train chung là chạy `mimickit/run.py`, nạp config cho engine, environment và agent, rồi train hoặc test model theo `--mode`.
- `Isaac Lab` đóng vai trò là một backend engine trong kiến trúc của repo, chịu trách nhiệm phần simulator cho environment chứ không phải toàn bộ logic của MimicKit.

---

## Bước 2. Đọc ba file config mẫu quan trọng nhất

## 1. `data/engines/isaac_lab_engine.yaml`

```yaml
engine_name: "isaac_lab"
control_mode: "pos"
control_freq: 30
sim_freq: 120
env_spacing: 5
```

### Ý nghĩa các key

- `engine_name`: tên backend simulator mà builder sẽ khởi tạo. Ở đây là `Isaac Lab`.
- `control_mode`: cách policy điều khiển robot. `pos` nghĩa là action được hiểu như target position cho joint.
- `control_freq`: tần số agent/env ra quyết định, tức số lần áp action mỗi giây.
- `sim_freq`: tần số physics simulation thật của engine, tức số physics step mỗi giây.
- `env_spacing`: khoảng cách đặt giữa các environment khi clone nhiều env song song.

### Trả lời câu hỏi

- Vì `sim_freq` nên chia hết cho `control_freq` để mỗi action được giữ trong một số nguyên physics steps, tránh lệch nhịp control.
- Nếu `control_freq = 30` và `sim_freq = 120` thì một action được giữ trong `120 / 30 = 4` physics steps.
- Ví dụ theo timeline logic:
  
  `control_freq = 30` nghĩa là policy chỉ được đổi action mỗi `1/30` giây.
  
  `sim_freq = 120` nghĩa là physics chạy mỗi `1/120` giây.
  
  Vì `1/30 = 4 x (1/120)`, nên trong thời gian policy giữ nguyên một quyết định, physics sẽ chạy 4 nhịp.
  
  Timeline:
  
  `physics step 1`: policy ra `action A`
  
  `physics step 2`: vẫn giữ `action A`
  
  `physics step 3`: vẫn giữ `action A`
  
  `physics step 4`: vẫn giữ `action A`
  
  `physics step 5`: policy mới được đổi sang `action B`
  
  Nghĩa là `control` quyết định lúc nào được đổi lệnh, còn `physics` quyết định thế giới được mô phỏng mịn tới mức nào.

## 2. `data/envs/deepmimic_humanoid_env.yaml`

### Ý nghĩa các key

- `env_name` mặc định: `"deepmimic"`; tên environment logic sẽ được tạo.
- `char_file` mặc định: `"data/assets/humanoid/humanoid.xml"`; file asset/mô hình nhân vật dùng cho env.
- `camera_mode` mặc định: `"track"`; chế độ camera khi visualize, ví dụ bám theo nhân vật.
- `episode_length` mặc định: `10.0`; độ dài tối đa của một episode, tính theo giây.
- `global_obs` mặc định: `True`; observation có dùng hệ quy chiếu/global features hay không.
- `root_height_obs` mặc định: `True`; có đưa chiều cao root vào observation hay không.
- `pose_termination` mặc định: `True`; có terminate khi pose lệch quá xa motion reference hay không.
- `pose_termination_dist` mặc định: `1.0`; ngưỡng khoảng cách để kích hoạt pose termination, đơn vị mét.
- `enable_phase_obs` mặc định: `False`; có thêm phase của motion vào observation hay không.
- `enable_tar_obs` mặc định: `True`; có thêm target/reference future observation vào observation hay không.
- `num_phase_encoding` mặc định: `4`; số chiều encoding cho phase nếu dùng phase observation.
- `tar_obs_steps` mặc định: `[1, 2, 3]`; các mốc step tương lai lấy ra để tạo target observation.
- `rand_reset` mặc định: `True`; reset ngẫu nhiên theo các thời điểm khác nhau trong motion thay vì luôn bắt đầu từ đầu clip.
- `log_tracking_error` mặc định: `False`; có log lỗi tracking để debug/phân tích hay không.
- `ref_char_offset` mặc định: `[2.0, 0.0, 0.0]`; độ lệch vị trí của reference character khi visualize song song với nhân vật chính.
- `init_pose` mặc định: `[0, 0, 0.882416, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5708, 0, 0, 0, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`; pose khởi tạo mặc định của character khi reset theo tư thế cố định.
- `enable_early_termination` mặc định: `True`; cho phép dừng episode sớm khi có điều kiện fail.
- `key_bodies` mặc định: `["head", "right_hand", "left_hand", "right_foot", "left_foot"]`; các body quan trọng dùng để so khớp motion/reward.
- `contact_bodies` mặc định: `["right_foot", "left_foot"]`; các body được phép chạm đất hoặc được dùng trong logic contact/termination.
- `motion_file` mặc định: `"data/motions/humanoid/humanoid_spinkick.pkl"`; nguồn motion reference, có thể là clip `.pkl` hoặc dataset `.yaml`.
- `joint_err_w` mặc định: `[1.0, 0.6, 0.6, 0.4, 0.0, 0.6, 0.4, 0.0, 1.0, 0.6, 0.4, 1.0, 0.6, 0.4]`; trọng số lỗi cho từng joint khi tính reward pose matching.
- `reward_pose_w` mặc định: `0.5`; trọng số reward cho pose.
- `reward_vel_w` mặc định: `0.1`; trọng số reward cho joint velocity.
- `reward_root_pose_w` mặc định: `0.15`; trọng số reward cho root pose.
- `reward_root_vel_w` mặc định: `0.1`; trọng số reward cho root velocity.
- `reward_key_pos_w` mặc định: `0.15`; trọng số reward cho key body positions.
- `reward_pose_scale` mặc định: `0.25`; scale chuyển lỗi pose thành reward.
- `reward_vel_scale` mặc định: `0.01`; scale chuyển lỗi velocity thành reward.
- `reward_root_pose_scale` mặc định: `5.0`; scale cho root pose reward.
- `reward_root_vel_scale` mặc định: `1.0`; scale cho root velocity reward.
- `reward_key_pos_scale` mặc định: `10.0`; scale cho key position reward.

### Phân loại nhanh theo vai trò

- Ảnh hưởng simulator/character: `char_file`, `camera_mode`, `ref_char_offset`.
- Ảnh hưởng reset: `rand_reset`, `init_pose`, `motion_file`.
- Ảnh hưởng observation: `global_obs`, `root_height_obs`, `enable_phase_obs`, `num_phase_encoding`, `enable_tar_obs`, `tar_obs_steps`.
- Ảnh hưởng reward: `key_bodies`, `joint_err_w`, toàn bộ các key `reward_*`.
- Ảnh hưởng termination/episode: `episode_length`, `pose_termination`, `pose_termination_dist`, `enable_early_termination`, `contact_bodies`.
- Ảnh hưởng logging/debug: `log_tracking_error`.

## 3. `data/agents/deepmimic_humanoid_ppo_agent.yaml`

### Ý nghĩa các key

- `agent_name` mặc định: `"PPO"`; tên thuật toán agent.

### Nhóm `model`

- `actor_net` mặc định: `"fc_2layers_1024units"`; kiến trúc mạng actor.
- `actor_init_output_scale` mặc định: `0.01`; scale khởi tạo lớp output của actor để action ban đầu nhỏ và ổn định hơn.
- `actor_std_type` mặc định: `"FIXED"`; cách quản lý độ lệch chuẩn của policy, ở đây std cố định.
- `action_std` mặc định: `0.05`; độ lệch chuẩn của policy stochastic, điều khiển mức độ exploration.
- `critic_net` mặc định: `"fc_2layers_1024units"`; kiến trúc mạng critic.

### Nhóm optimizer

- `actor_optimizer.type` mặc định: `"SGD"`; loại optimizer cho actor.
- `actor_optimizer.learning_rate` mặc định: `1e-4`; learning rate của actor.
- `critic_optimizer.type` mặc định: `"SGD"`; loại optimizer cho critic.
- `critic_optimizer.learning_rate` mặc định: `1e-4`; learning rate của critic.

### Nhóm train loop

- `discount` mặc định: `0.99`; hệ số discount reward tương lai.
- `steps_per_iter` mặc định: `32`; số bước rollout/environment steps thu thập trước mỗi vòng update.
- `iters_per_output` mặc định: `100`; bao nhiêu iteration thì log/in kết quả một lần.
- `test_episodes` mặc định: `32`; số episode dùng khi evaluate/test.
- `normalizer_samples` mặc định: `100000000`; số sample dùng để thống kê chuẩn hóa observation.

### Nhóm PPO update

- `actor_epochs` mặc định: `5`; số epoch update actor trên cùng một batch rollout.
- `actor_batch_size` mặc định: `4`; số minibatch cho actor hoặc kích thước chia batch theo cách code repo dùng.
- `critic_epochs` mặc định: `2`; số epoch update critic trên cùng một batch rollout.
- `critic_batch_size` mặc định: `2`; số minibatch cho critic hoặc kích thước chia batch theo cách code repo dùng.
- `td_lambda` mặc định: `0.95`; hệ số lambda cho TD/GAE.
- `ppo_clip_ratio` mặc định: `0.2`; ngưỡng clip của PPO objective.
- `norm_adv_clip` mặc định: `4.0`; ngưỡng clip cho normalized advantage.
- `action_bound_weight` mặc định: `10.0`; trọng số penalty giữ action trong miền hợp lý.
- `action_entropy_weight` mặc định: `0.0`; trọng số entropy bonus để khuyến khích exploration.
- `action_reg_weight` mặc định: `0.0`; trọng số regularization lên action.

### Trả lời câu hỏi

- `steps_per_iter` là số bước dữ liệu agent phải chạy và thu thập trước khi thực hiện một vòng học/update.
- `actor_epochs`, `critic_epochs` là số lần quét lại batch dữ liệu đã thu thập để update actor/critic; khác với `steps_per_iter` vì `steps_per_iter` là giai đoạn thu thập dữ liệu, còn `*_epochs` là giai đoạn học trên dữ liệu đó.
- `action_std` quyết định độ ngẫu nhiên của action sampling: std lớn thì exploration mạnh hơn, std nhỏ thì policy hành động ổn định và ít khám phá hơn.
- Ví dụ cho `steps_per_iter`: nếu mỗi env chạy 32 bước rồi mới update, thì agent sẽ quan sát 32 bước chuyển động, reward, done rồi mới bắt đầu tối ưu actor/critic.
- Ví dụ cho `actor_epochs` và `critic_epochs`: sau khi thu thập xong batch 32 bước đó, actor có thể học lại trên cùng batch 5 lần và critic học 2 lần; tức là một batch dữ liệu được dùng nhiều lượt để update thay vì thu thập xong là bỏ ngay.
- Ví dụ cho `action_std`: nếu policy dự đoán góc khớp nên là `0.5`, thì với `action_std` nhỏ agent thường lấy action gần `0.5`; còn nếu `action_std` lớn thì action có thể dao động mạnh hơn quanh `0.5`, giúp thử nhiều chuyển động hơn lúc đầu train.

### Phân biệt `step rollout`, `iteration`, `episode`, `batch`, `epoch`

- `step rollout`: một bước tương tác env của agent. Ở mỗi step, agent đọc observation, chọn action, env chạy tiếp một nhịp control rồi trả về observation mới, reward và done.
- `iteration`: một vòng train lớn gồm 2 phần: thu thập dữ liệu rollout rồi dùng chính dữ liệu đó để update model. Trong file này, `steps_per_iter = 32` nghĩa là mỗi iteration sẽ đi thu thập 32 rollout steps trước.
- `episode`: một quãng chạy hoàn chỉnh từ lúc reset env đến khi env báo `done` hoặc chạm `episode_length`. Một iteration có thể chứa nhiều episode ngắn, hoặc chỉ là một đoạn cắt ra từ episode dài đang chạy dở.
- `batch`: cục dữ liệu được gom lại để đưa vào học một lần. Sau rollout, toàn bộ dữ liệu thu được sẽ thành batch lớn; rồi batch lớn này có thể bị chia tiếp thành các batch nhỏ hơn khi update actor/critic.
- `epoch`: một lượt quét hết qua batch dữ liệu để update model. Nếu `actor_epochs = 5` thì cùng một batch rollout đó actor sẽ học đi học lại 5 lượt.


### Flow train thật khi `steps_per_iter = 32`

- Bước 1: agent chưa học ngay, mà đi chạy env trước để thu thập dữ liệu.
- Bước 2: mỗi env chạy `32 step rollout`, nên sau bước này ta có một cục dữ liệu lớn của iteration hiện tại.
- Bước 3: chỉ sau khi gom đủ 32 step thì PPO mới bắt đầu pha update model.
- Bước 4: lúc update, cục dữ liệu 32 step này được chia nhỏ thành các `mini-batch` để train dần, chứ không phải học từng step lẻ.
- Bước 5: khi học xong hết các mini-batch của iteration này, nếu còn `epoch` thì quay lại dùng lại chính cục dữ liệu đó để học thêm lượt nữa.
- Bước 6: học xong đủ số `epoch` thì iteration kết thúc, lúc đó agent mới quay lại env để thu thập 32 step mới cho iteration tiếp theo.


### Ví dụ rất cụ thể

- Giả sử đơn giản ta có `32 step` dữ liệu và chia thành `4 mini-batch`.
- `epoch 1`:
  mini-batch 1 -> mini-batch 2 -> mini-batch 3 -> mini-batch 4
- `epoch 2`:
  lại dùng chính dữ liệu cũ và quét tiếp
  mini-batch 1 -> mini-batch 2 -> mini-batch 3 -> mini-batch 4
- Cứ như vậy cho đến hết `actor_epochs` hoặc `critic_epochs`.
- Sau đó mới quay lại env để thu thập `32 step` mới.

### Cách hình dung ngắn

- `32 step rollout` là giai đoạn đi lấy dữ liệu.
- `chia batch` là giai đoạn cắt dữ liệu đó ra để học.
- `epoch` là học đi học lại trên cùng dữ liệu đã lấy.
- PPO không vừa lấy từng step xong là train ngay từng step đó.

### Cách nhớ ngắn

- `step rollout`: 1 bước tương tác với env
- `episode`: 1 lần sống từ reset đến done
- `iteration`: 1 vòng thu thập rồi update
- `batch`: dữ liệu đem ra học
- `epoch`: 1 lượt học quét qua batch

### Tóm lại ngắn gọn

- File engine quyết định simulator backend chạy kiểu gì.
- File env quyết định bài toán imitation cụ thể, observation, reset, reward và termination.
- File agent quyết định cách PPO biểu diễn policy/value và cách tối ưu trong lúc train.

---

## Bước 3. Mở file arg preset để hiểu cách người repo chạy thực tế

### File đang xem

- `args/deepmimic_humanoid_ppo_args.txt`

### Nội dung preset

```txt
--num_envs 4096

--engine_config data/engines/isaac_gym_engine.yaml
--env_config data/envs/deepmimic_humanoid_env.yaml
--agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml

--out_dir output/
```

### Ý nghĩa các dòng

- `--num_envs 4096`: chạy 4096 environment song song, phù hợp cho train tốc độ cao trên backend hỗ trợ vectorized simulation.
- `--engine_config data/engines/isaac_gym_engine.yaml`: chọn backend simulator là `Isaac Gym`.
- `--env_config data/envs/deepmimic_humanoid_env.yaml`: chọn bài toán environment là `DeepMimic humanoid`.
- `--agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml`: chọn agent học là `PPO`.
- `--out_dir output/`: nơi lưu model, log và kết quả train.

### Điều cần hiểu từ preset này

- File `arg` là cách repo gom sẵn các tham số CLI thường dùng để không phải gõ dài mỗi lần chạy.
- Preset này ghép đúng 3 lớp cấu hình của repo: `engine + env + agent`.
- `env_config` và `agent_config` đang đúng với case ta đang học: `DeepMimic humanoid + PPO`.
- Nhưng `engine_config` trong preset mặc định là `Isaac Gym`, không phải `Isaac Lab`.
- Nếu muốn học hoặc chạy theo trục `Isaac Lab`, chỉ cần đổi sang `data/engines/isaac_lab_engine.yaml`.

### So sánh nhanh với Bước 2

- Engine config trong Bước 2 đang đọc là `isaac_lab_engine.yaml`, còn preset này dùng `isaac_gym_engine.yaml`.
- Env config giữ nguyên: cùng dùng `deepmimic_humanoid_env.yaml`.
- Agent config giữ nguyên: cùng dùng `deepmimic_humanoid_ppo_agent.yaml`.
- Nghĩa là cùng một bài toán và cùng một agent có thể chạy trên nhiều engine khác nhau, miễn engine backend đó được hỗ trợ.

### Kết luận ngắn

- Bước 3 cho thấy cách repo được chạy ngoài thực tế là lấy một preset args để nối 3 file config lại.
- Đây cũng là bằng chứng rõ nhất rằng kiến trúc repo tách riêng `engine`, `environment` và `agent`.
