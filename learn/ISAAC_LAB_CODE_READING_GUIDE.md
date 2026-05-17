# MimicKit + Isaac Lab: Hướng Dẫn Đọc Code Từng Bước Siêu Chi Tiết

## Mục tiêu của tài liệu này

Tài liệu này dành cho việc đọc source code theo lộ trình, không phải chỉ để “biết repo có gì”.

Mục tiêu là sau khi đi hết tài liệu, bạn sẽ tự trả lời được:

1. `MimicKit` tổ chức code theo những lớp nào.
2. Luồng chạy từ lệnh `python mimickit/run.py ...` đi qua những file nào.
3. Isaac Lab nằm ở đâu trong kiến trúc tổng thể.
4. DeepMimic environment được xây như thế nào.
5. Motion data, character model, observation, reward, done, agent PPO liên kết với nhau ra sao.
6. Muốn debug hoặc tự sửa một bài toán imitation trong repo thì nên bắt đầu từ đâu.

Tài liệu này lấy trục chính là case quan trọng nhất để học repo:

- Engine: `Isaac Lab`
- Environment: `DeepMimic`
- Agent: `PPO`
- Character: `humanoid`

Đây là đường học tốt nhất vì nó chạm hầu hết các thành phần cốt lõi của codebase.

---

## Tư duy tổng thể trước khi đọc

Đừng đọc repo này theo kiểu mở từng file ngẫu nhiên.

Cách đúng là xem nó như pipeline 6 lớp:

1. `Entry point`
   `mimickit/run.py`
2. `Config dispatch`
   YAML trong `data/engines/`, `data/envs/`, `data/agents/`
3. `Builder layer`
   `env_builder.py`, `agent_builder.py`, `engine_builder.py`
4. `Simulation abstraction`
   `Engine`, `SimEnv`, `CharEnv`, `DeepMimicEnv`
5. `Motion + kinematics`
   `motion.py`, `motion_lib.py`, `kin_char_model.py`
6. `Learning`
   `BaseAgent`, `PPOAgent`, model, optimizer, replay/experience buffer

Isaac Lab chỉ là một phần của lớp số 4.

Đây là ý rất quan trọng:

- `MimicKit` không được thiết kế như “repo Isaac Lab”.
- Nó được thiết kế như “motion imitation framework có nhiều simulator backend”.
- Isaac Lab là một backend implementation thông qua `IsaacLabEngine`.

Nói ngắn gọn:

- Logic task nằm chủ yếu ở `envs/`
- Logic RL nằm chủ yếu ở `learning/`
- Logic simulator-specific nằm ở `engines/`

Nếu nắm được điều này sớm, bạn sẽ đỡ bị lạc.

---

## Bản đồ thư mục nên hiểu trước

Các thư mục quan trọng:

- `mimickit/run.py`
  Điểm vào chính để train/test.
- `mimickit/envs/`
  Chứa logic task/environment.
- `mimickit/engines/`
  Chứa abstraction của simulator và các backend như Isaac Gym, Isaac Lab, Newton.
- `mimickit/learning/`
  Chứa agent RL và model.
- `mimickit/anim/`
  Chứa motion representation, motion loading, kinematic character model.
- `mimickit/util/`
  Chứa công cụ phụ trợ như parsing args, logging, torch utils, multiprocessing.
- `data/envs/`
  YAML cấu hình environment.
- `data/engines/`
  YAML cấu hình backend engine.
- `data/agents/`
  YAML cấu hình thuật toán học.
- `data/assets/`
  Character assets như MJCF, URDF, USD.
- `data/motions/`
  Motion clip `.pkl`.
- `args/`
  Preset command-line arguments.
- `docs/`
  Tài liệu theo từng phương pháp.

Thứ tự hiểu thư mục:

1. `README.md`
2. `data/engines/`, `data/envs/`, `data/agents/`
3. `mimickit/run.py`
4. `mimickit/envs/`
5. `mimickit/engines/`
6. `mimickit/anim/`
7. `mimickit/learning/`

Lý do:

- Config cho bạn biết repo “muốn chạy cái gì”.
- `run.py` cho bạn biết repo “thực sự khởi động như thế nào”.
- Sau đó mới đọc class implementation cụ thể.

---

## Lộ trình học đề xuất

Nếu bạn muốn hiểu repo từng bước và không bị ngợp, hãy đi đúng thứ tự này:

1. Hiểu repo ở mức command line và config.
2. Hiểu luồng `run.py`.
3. Hiểu cơ chế `builder`.
4. Hiểu abstraction `Engine`.
5. Hiểu `SimEnv`.
6. Hiểu `CharEnv`.
7. Hiểu `DeepMimicEnv`.
8. Hiểu `motion.py` và `motion_lib.py`.
9. Hiểu `kin_char_model.py`.
10. Hiểu `BaseAgent`.
11. Hiểu `PPOAgent`.
12. Quay lại `IsaacLabEngine` để nối tất cả lại.

Nghe có vẻ lạ vì Isaac Lab chưa đọc ngay, nhưng đây là thứ tự đúng.

Nếu bạn đọc `isaac_lab_engine.py` quá sớm, bạn sẽ thấy rất nhiều API simulator nhưng chưa hiểu chúng phục vụ logic nào.

---

## Giai đoạn 1: Nắm cách chạy repo từ ngoài vào

### Bước 1. Đọc `README.md`

File:

- `README.md`

Bạn cần trả lời được:

1. Repo này dùng để làm gì.
2. Nó hỗ trợ những phương pháp imitation nào.
3. Nó hỗ trợ những engine nào.
4. Cách train/test tổng quát ra sao.
5. Motion data được biểu diễn như thế nào.

Những ý quan trọng phải note lại:

- Repo hỗ trợ nhiều engine: Isaac Gym, Isaac Lab, Newton.
- Engine được chọn bằng `--engine_config`.
- Environment được chọn bằng `--env_config`.
- Agent được chọn bằng `--agent_config`.
- Motion có thể là một clip `.pkl` hoặc dataset `.yaml`.

Sau khi đọc xong, bạn nên tự viết lại bằng 3 câu:

- Một câu mô tả repo.
- Một câu mô tả pipeline train.
- Một câu mô tả vai trò của Isaac Lab.

### Bước 2. Đọc ba file config mẫu quan trọng nhất

Files:

- `data/engines/isaac_lab_engine.yaml`
- `data/envs/deepmimic_humanoid_env.yaml`
- `data/agents/deepmimic_humanoid_ppo_agent.yaml`

Đây là bộ ba cấu hình tốt nhất để học repo.

Bạn cần đọc theo cách sau:

#### 2.1. Engine config

`data/engines/isaac_lab_engine.yaml`

Các trường chính:

- `engine_name: "isaac_lab"`
- `control_mode: "pos"`
- `control_freq: 30`
- `sim_freq: 120`
- `env_spacing: 5`

Ý nghĩa:

- `engine_name` quyết định builder sẽ tạo class engine nào.
- `control_mode` quyết định action của policy sẽ được engine áp vào simulator theo kiểu nào.
- `control_freq` là tần số decision của policy/env.
- `sim_freq` là tần số physics step thật bên simulator.
- `env_spacing` là khoảng cách giữa các env clone trong world.

Câu hỏi bạn phải tự trả lời:

1. Tại sao `sim_freq` phải chia hết cho `control_freq`?
2. Nếu `control_freq = 30` và `sim_freq = 120` thì một action được giữ trong bao nhiêu physics steps?

#### 2.2. Env config

`data/envs/deepmimic_humanoid_env.yaml`

Các nhóm biến quan trọng:

- Character:
  `char_file`, `init_pose`
- Episode:
  `episode_length`, `enable_early_termination`
- Observation:
  `global_obs`, `root_height_obs`, `enable_phase_obs`, `enable_tar_obs`, `tar_obs_steps`
- Motion:
  `motion_file`, `rand_reset`
- Visualization:
  `camera_mode`, `ref_char_offset`, `visualize_ref_char`
- Reward:
  `reward_*_w`, `reward_*_scale`
- Contact/termination:
  `contact_bodies`, `pose_termination`, `pose_termination_dist`

Bạn cần phân loại từng field:

1. Field nào ảnh hưởng simulator.
2. Field nào ảnh hưởng reset.
3. Field nào ảnh hưởng observation.
4. Field nào ảnh hưởng reward.
5. Field nào ảnh hưởng termination.

#### 2.3. Agent config

`data/agents/deepmimic_humanoid_ppo_agent.yaml`

Các nhóm biến:

- Kiến trúc actor/critic.
- Optimizer.
- Discount / rollout.
- PPO hyperparameters.
- Normalizer / logging.

Bạn cần tự trả lời:

1. `steps_per_iter` dùng để làm gì?
2. `actor_epochs`, `critic_epochs` khác `steps_per_iter` ở đâu?
3. `action_std` ảnh hưởng exploration kiểu gì?

### Bước 3. Mở file arg preset để hiểu cách người repo chạy thực tế

Thư mục:

- `args/`

Bạn nên mở file tương ứng với case đang học, ví dụ preset DeepMimic humanoid.

Mục đích:

- So sánh preset CLI với 3 YAML phía trên.
- Hiểu biến nào thường được override từ command line.
- Biết “cấu hình thực chiến” chứ không chỉ cấu hình lý thuyết.

---

## Giai đoạn 2: Hiểu entry point và cơ chế dựng object

### Bước 4. Đọc `mimickit/run.py`

File:

- `mimickit/run.py`

Đây là file quan trọng nhất để bắt đầu đọc source.

Bạn phải lần theo đúng các hàm này:

1. `main(argv)`
2. `load_args(argv)`
3. `run(rank, num_procs, device, master_port, args)`
4. `build_env(args, ...)`
5. `build_agent(args, ...)`
6. `train(...)` hoặc `test(...)`

### Điều cần hiểu trong `run.py`

#### 4.1. `load_args`

Nó load args từ command line trước, sau đó nếu có `arg_file` thì load tiếp file đó.

Điểm tinh tế:

- `arg_file` không thay thế command line hoàn toàn.
- Nó là cơ chế “nạp sẵn preset”, rồi bạn vẫn có thể override bằng CLI.

#### 4.2. `run`

Đây là flow cốt lõi:

1. Parse mode, num_envs, visualize, logger, model_file, out_dir.
2. Init multiprocessing utilities.
3. Set random seed.
4. Build env.
5. Build agent.
6. Nếu có `model_file` thì load checkpoint.
7. Nếu `mode == train` thì train.
8. Nếu `mode == test` thì test.

Bạn phải vẽ được sơ đồ:

`args -> env_builder -> env`

`args -> agent_builder -> agent`

`agent <-> env <-> engine <-> simulator`

#### 4.3. Multiprocessing

`run.py` hỗ trợ nhiều device:

- `cpu`
- `cuda:0`, `cuda:1`, ...

Điểm cần hiểu:

- Nó spawn nhiều process bằng `torch.multiprocessing`.
- Mỗi process chạy `run(...)`.
- Đây là data collection / training parallel theo nhiều process, không phải chỉ đơn giản là vectorized env trong một process.

Nếu bạn chưa cần distributed training, chỉ cần note:

- Repo có 2 tầng song song hóa:
  1. nhiều env trong engine
  2. nhiều process trên nhiều device

### Bài tập nhỏ sau bước 4

Không cần sửa code, chỉ cần trả lời:

1. Nếu đổi `--mode test` thì flow đổi ở đâu?
2. `env` được tạo trước hay `agent` được tạo trước?
3. `agent` có nhận `env` trong constructor không?
4. Chỗ nào save lại config vào `out_dir`?

---

## Giai đoạn 3: Hiểu cơ chế builder

### Bước 5. Đọc `mimickit/envs/env_builder.py`

File:

- `mimickit/envs/env_builder.py`

Vai trò:

- Load YAML.
- Xác định `env_name`.
- Import đúng class env rồi khởi tạo.

Điểm quan trọng:

- Repo dùng pattern “string trong YAML -> class Python”.
- Đây là nơi dispatch chính của environment.

Bạn cần nắm:

1. `load_configs(env_file, engine_file)`
2. `override_engine_config(...)`
3. `build_env(...)`

Ý hay ở đây:

- Env config có thể override engine config.
- Nghĩa là simulator backend có config chung, nhưng từng env vẫn có thể ép sửa vài tham số engine.

### Bước 6. Đọc `mimickit/learning/agent_builder.py`

File:

- `mimickit/learning/agent_builder.py`

Vai trò:

- Load `agent_config`.
- Dựa vào `agent_name` để tạo `PPO`, `AWR`, `AMP`, `ASE`, `ADD`, `LCP`, ...

Điểm phải để ý:

- `agent_builder` gọi `agent.calc_num_params()`.
- Nghĩa là sau khi build, agent đã đủ hoàn chỉnh để đếm tham số model.

### Bước 7. Đọc `mimickit/engines/engine_builder.py`

File:

- `mimickit/engines/engine_builder.py`

Vai trò:

- Nhìn `engine_name` rồi chọn:
  - `IsaacGymEngine`
  - `IsaacLabEngine`
  - `NewtonEngine`

Đây là điểm nối trực tiếp giữa config và simulator backend.

### Câu hỏi kiểm tra sau builder layer

1. `env_name`, `agent_name`, `engine_name` được định nghĩa ở đâu?
2. Nếu muốn thêm env mới, cần sửa ít nhất file nào?
3. Nếu muốn thêm engine mới, cần sửa ít nhất file nào?

---

## Giai đoạn 4: Hiểu abstraction của simulator

### Bước 8. Đọc `mimickit/engines/engine.py`

File:

- `mimickit/engines/engine.py`

Đây là file bắt buộc phải hiểu trước khi đọc Isaac Lab engine.

Vai trò của file:

- Định nghĩa interface chung mà mọi simulator backend phải cung cấp.

Bạn hãy đọc theo nhóm method:

#### 8.1. Nhóm tạo scene

- `create_env`
- `create_obj`
- `initialize_sim`

#### 8.2. Nhóm step/control

- `set_cmd`
- `step`
- `get_timestep`
- `get_control_mode`

#### 8.3. Nhóm camera/render/UI

- `set_camera_pose`
- `get_camera_pos`
- `get_camera_dir`
- `render`
- `register_keyboard_callback`

#### 8.4. Nhóm read state

- `get_root_pos`
- `get_root_rot`
- `get_root_vel`
- `get_dof_pos`
- `get_body_pos`
- `get_ground_contact_forces`
- ...

#### 8.5. Nhóm write state

- `set_root_pos`
- `set_root_rot`
- `set_dof_pos`
- `set_body_vel`
- ...

#### 8.6. Nhóm metadata

- `get_obj_num_dofs`
- `get_obj_body_names`
- `find_obj_body_id`
- `get_obj_dof_limits`
- `get_obj_torque_limits`
- `calc_obj_mass`

### Tại sao file này cực kỳ quan trọng

Toàn bộ `envs/` không muốn biết chi tiết Isaac Lab API.

`envs/` chỉ muốn biết:

- lấy state kiểu nào
- set state kiểu nào
- apply action kiểu nào
- step simulator kiểu nào

Vì vậy `Engine` chính là lớp chống phụ thuộc trực tiếp vào backend.

Nếu bạn hiểu `engine.py`, bạn sẽ đọc `CharEnv` và `DeepMimicEnv` nhanh hơn rất nhiều.

---

## Giai đoạn 5: Hiểu khung environment tổng quát

### Bước 9. Đọc `mimickit/envs/base_env.py`

File:

- `mimickit/envs/base_env.py`

Vai trò:

- Interface tối thiểu cho environment theo phong cách RL.

Các enum cần ghi nhớ:

- `EnvMode.TRAIN`
- `EnvMode.TEST`
- `DoneFlags.NULL`
- `DoneFlags.FAIL`
- `DoneFlags.SUCC`
- `DoneFlags.TIME`

Điểm quan trọng:

- Repo không dùng `done` kiểu boolean thuần.
- Nó encode loại termination bằng integer flag.

Điều này ảnh hưởng trực tiếp tới cách agent tính return bootstrapping.

### Bước 10. Đọc `mimickit/envs/sim_env.py`

File:

- `mimickit/envs/sim_env.py`

Đây là base class rất quan trọng cho mọi env dựa trên simulator.

Hãy đọc constructor `__init__` thật chậm:

1. Lấy `episode_length`.
2. Build engine.
3. Build envs trong simulator.
4. `engine.initialize_sim()`.
5. Build action space.
6. Build tensors/buffers.
7. Nếu visualize thì setup camera và GUI.

### Ý nghĩa lớn của `SimEnv`

`SimEnv` giải quyết phần khung lặp chuẩn:

- reset
- step
- pre physics
- physics
- post physics
- render
- buffer hóa observation/reward/done/time

### Các hàm bạn phải hiểu kỹ

#### 10.1. `reset`

Flow:

1. Chọn `env_ids` cần reset.
2. Gọi `_reset_envs`.
3. Update obs.
4. Update info.
5. Trả về `obs, info`.

#### 10.2. `step`

Flow:

1. `_pre_physics_step(action)`
2. `_physics_step()`
3. `_post_physics_step()`
4. Nếu visualize thì `_render()`
5. Trả về `obs, reward, done, info`

Đây chính là “xương sống runtime” của environment.

#### 10.3. `_post_physics_step`

Flow:

1. Update time
2. Update misc
3. Update observations
4. Update info
5. Update reward
6. Update done

Bạn phải hiểu rất rõ thứ tự này, vì:

- reward dùng state sau step
- done dùng state sau step
- ref motion trong DeepMimic cũng được update ở pha misc

#### 10.4. `_build_data_buffers`

Các buffer quan trọng:

- `_reward_buf`
- `_done_buf`
- `_timestep_buf`
- `_time_buf`
- `_obs_buf`
- `_info`

### Câu hỏi kiểm tra sau `SimEnv`

1. Observation được tính trước hay sau physics step?
2. Buffer thời gian nằm ở đâu?
3. Render có nằm trong logic step không?
4. `SimEnv` biết gì về humanoid hay motion không?

Đáp án đúng cho câu 4 là: không, nó hoàn toàn generic.

---

## Giai đoạn 6: Hiểu character environment

### Bước 11. Đọc `mimickit/envs/char_env.py`

File:

- `mimickit/envs/char_env.py`

Đây là lớp chuyển từ “simulator env tổng quát” sang “env có character articulated”.

### Vai trò của `CharEnv`

Nó thêm 5 thứ lớn:

1. Load character kinematic model.
2. Spawn articulated character vào engine.
3. Xây action space từ giới hạn joint / torque / control mode.
4. Định nghĩa observation cơ bản cho character.
5. Định nghĩa reset/action cho character.

### Cách đọc `CharEnv`

#### 11.1. Constructor

Các field đáng chú ý:

- `_global_obs`
- `_root_height_obs`
- `_zero_center_action`

Sau đó `super().__init__` gọi lên `SimEnv`, tức là trong lúc constructor chạy, môi trường đã thực sự build xong.

#### 11.2. `_build_envs`

Flow:

1. Load kinematic character model từ `char_file`.
2. Parse `init_pose`.
3. Với mỗi env:
   - `engine.create_env()`
   - `_build_env(env_id, env_config)`

Đây là chỗ character được tạo lặp lại cho từng parallel env.

#### 11.3. `_build_kin_char_model`

Điểm rất đáng chú ý:

- `.xml` -> `MJCFCharModel`
- `.urdf` -> `URDFCharModel`
- `.usd` -> `USDCharModel`

Điều này cho thấy:

- Repo tách “mô hình kinematic” khỏi simulator backend.
- Cùng một asset format có thể được phân tích để dùng cho observation/reward/forward kinematics.

#### 11.4. `_build_character`

Gọi:

- `engine.create_obj(...)`

với:

- `obj_type = articulated`
- `asset_file = char_file`
- `start_pos`, `start_rot`

Tức là `CharEnv` không tự biết cách load MJCF/USD vào Isaac Lab.
Việc đó là nhiệm vụ của engine.

#### 11.5. `_build_action_space`

Đây là đoạn rất đáng học.

Action space phụ thuộc vào `control_mode`:

- `none`
- `vel`
- `torque`
- `pos`
- `pd_explicit`

Nghĩa là cùng một character, action vector có thể mang nghĩa hoàn toàn khác tùy engine config.

Đây là một ý rất quan trọng khi debug:

- model output shape có thể giống nhau
- nhưng semantics của action khác nhau

#### 11.6. `_compute_obs`

`CharEnv` lấy từ engine:

- root pose
- root velocity
- dof pos/vel
- body positions

Sau đó:

- đổi `dof_pos` thành `joint_rot` bằng kinematic model
- build observation tensor bằng `compute_char_obs(...)`

Điểm cần nắm:

- Env logic không đọc trực tiếp internal Isaac Lab tensors.
- Nó luôn qua engine abstraction.

#### 11.7. `_reset_envs` và `_reset_char`

Reset character gồm:

- root pos / rot
- root vel / ang vel
- dof pos / vel
- body vel / ang vel

Sau đó `_reset_char_rigid_body_state` dùng forward kinematics để set lại body pose nhất quán.

Đây là chi tiết rất quan trọng:

- repo không chỉ set joint state
- nó còn đồng bộ rigid body state để simulator bắt đầu ở trạng thái hợp lệ

### Bước 12. Hiểu các hàm JIT ở cuối `char_env.py`

Các hàm như:

- `compute_char_obs`
- `compute_reward`
- `compute_done`

Vai trò:

- tách phần tensor math nặng ra thành các hàm `torch.jit.script`
- tăng tốc và giữ logic gọn hơn

Với `CharEnv`, reward/done rất đơn giản.

Mục đích của `CharEnv` không phải task thật, mà là lớp nền cho các env phức tạp hơn như `DeepMimicEnv`.

### Câu hỏi kiểm tra sau `CharEnv`

1. `CharEnv` khác `SimEnv` ở điểm nào lớn nhất?
2. Tại sao action bounds phải phụ thuộc `control_mode`?
3. Vì sao reset xong còn phải set body pose qua forward kinematics?
4. Observation của `CharEnv` đã chứa target motion chưa?

Đáp án câu 4 là chưa.

---

## Giai đoạn 7: Hiểu DeepMimic environment

### Bước 13. Đọc `mimickit/envs/deepmimic_env.py`

File:

- `mimickit/envs/deepmimic_env.py`

Đây là file quan trọng nhất nếu bạn muốn hiểu motion imitation trong repo.

### DeepMimicEnv thêm gì lên trên CharEnv

1. Motion library tham chiếu.
2. Reference state theo thời gian.
3. Reset theo motion sample.
4. Observation chứa phase/target information.
5. Reward tracking motion.
6. Early termination theo pose/contact/motion end.
7. Reference character để visualize.

### Cách đọc file này đúng thứ tự

#### 13.1. Constructor

Hãy chia config thành các nhóm:

- Termination:
  `enable_early_termination`, `pose_termination`, `pose_termination_dist`
- Observation:
  `enable_phase_obs`, `enable_tar_obs`, `tar_obs_steps`, `num_phase_encoding`
- Reset:
  `rand_reset`
- Visualization:
  `ref_char_offset`, `visualize_ref_char`
- Reward:
  toàn bộ `reward_*`

Mục tiêu của constructor là load các cờ logic, chưa phải làm math.

#### 13.2. `_build_envs`

Flow:

1. Khởi tạo list chứa ref chars.
2. Gọi `super()._build_envs(...)` để build nhân vật chính.
3. Load motion file bằng `_load_motions(motion_file)`.

Điểm hay:

- Build simulator objects trước.
- Load motion library sau.

#### 13.3. `_build_env`

Ngoài character chính, nó có thể build thêm `ref_character`.

Reference character:

- chỉ dùng để nhìn
- không phải nhân vật điều khiển
- được tô màu khác
- có thể `disable_motors=True`
- có thể `enable_self_collisions=False`

Khi xem demo, đó là lý do bạn thấy “nhân vật policy” và “nhân vật tham chiếu” cạnh nhau.

#### 13.4. `_build_sim_tensors`

Đây là đoạn rất quan trọng.

Nó tạo các tensor để giữ state tham chiếu:

- `_motion_ids`
- `_motion_time_offsets`
- `_ref_root_pos`
- `_ref_root_rot`
- `_ref_root_vel`
- `_ref_root_ang_vel`
- `_ref_body_pos`
- `_ref_body_rot`
- `_ref_joint_rot`
- `_ref_dof_pos`
- `_ref_dof_vel`

Đây là “trí nhớ target motion” của env.

Hãy hiểu rõ:

- State thật của character nằm trong engine / simulator.
- State tham chiếu của motion nằm trong các tensor `_ref_*`.

Reward và observation sẽ so sánh hai bên này.

#### 13.5. `_reset_ref_motion`

Flow:

1. Sample `motion_ids`, `motion_times`.
2. Gọi `motion_lib.calc_motion_frame(...)`.
3. Ghi kết quả vào các tensor `_ref_*`.
4. Dùng `forward_kinematics(...)` để suy ra body pose tham chiếu.
5. Convert joint rotation sang dof position.

Đây là một trong những hàm quan trọng nhất toàn repo.

Nếu bạn hiểu hàm này, bạn sẽ hiểu reset imitation hoạt động thế nào.

#### 13.6. `_ref_state_init`

Sau khi đã có reference state, env set luôn nhân vật policy vào state đó.

Nghĩa là:

- khi reset, nhân vật không bắt đầu từ pose ngẫu nhiên vô nghĩa
- nó bắt đầu từ một frame hợp lệ của motion reference

Đây là kỹ thuật quan trọng để imitation học ổn định hơn.

#### 13.7. `_update_ref_motion`

Mỗi step:

1. Lấy `motion_ids`
2. Tính `motion_times = env_time + motion_time_offsets`
3. Query `motion_lib`
4. Update toàn bộ `_ref_*`

Nghĩa là target motion luôn chạy đồng bộ theo thời gian episode.

#### 13.8. `_compute_obs`

Đây là hàm phải đọc rất kỹ.

Observation của DeepMimic có thể gồm:

- state hiện tại của character
- phase của motion
- target future observations
- key body positions

Đặc biệt:

- `enable_phase_obs` quyết định có nhét phase vào obs không
- `enable_tar_obs` quyết định có nhét future target observation không
- `tar_obs_steps` quyết định nhìn trước mấy bước

Đây là chỗ bạn nên ngồi vẽ shape tensor ra giấy.

Bạn nên tự hỏi:

1. Phần nào của obs đến từ simulator?
2. Phần nào đến từ motion reference?
3. Phần nào đã được đổi về local frame?

#### 13.9. `_update_reward`

Reward được tính bằng cách so sánh:

- root pose
- root velocity
- joint pose
- dof velocity
- key body positions

giữa:

- character thật
- character reference

Bạn cần nắm:

- reward không phải một số đơn lẻ đơn giản
- nó là tổ hợp nhiều term với `weight` và `scale`

Đây là nơi bạn nên trace kỹ từng tham số trong YAML.

#### 13.10. `_update_done`

Termination có thể do:

- hết thời gian episode
- sai pose quá xa
- tiếp xúc đất không mong muốn
- motion kết thúc đối với non-looping motion

Nếu một model học không ổn, đây là file phải kiểm tra đầu tiên cùng với reward.

### Câu hỏi kiểm tra sau `DeepMimicEnv`

1. Reference motion được lưu ở simulator hay tensor trong env?
2. Tại sao reset phải sample thời điểm motion?
3. Tại sao cần `ref_character`?
4. Reward chính của DeepMimic so sánh cái gì với cái gì?
5. `enable_tar_obs` khác `enable_phase_obs` ở đâu?

---

## Giai đoạn 8: Hiểu motion representation

### Bước 14. Đọc `mimickit/anim/motion.py`

File:

- `mimickit/anim/motion.py`

Đây là file nhỏ nhưng rất quan trọng.

Nó cho bạn biết định dạng motion clip của repo:

- `loop_mode`
- `fps`
- `frames`

Mỗi frame có dạng:

- `root position (3)`
- `root rotation expmap (3)`
- `joint dof`

Điểm phải nhớ:

- root rotation trong file motion ban đầu là exponential map
- trong runtime, repo thường đổi sang quaternion để tính toán

### Bước 15. Đọc `mimickit/anim/motion_lib.py`

File:

- `mimickit/anim/motion_lib.py`

Đây là file cực kỳ quan trọng cho imitation.

### Vai trò của `MotionLib`

1. Load một motion clip hoặc motion dataset.
2. Chuyển frame data sang tensor runtime.
3. Tính phase.
4. Interpolate frame theo thời gian liên tục.
5. Tính root velocity, angular velocity, dof velocity.
6. Hỗ trợ sample motion cho reset.

### Những hàm bắt buộc phải hiểu

#### 15.1. `extract_pose_data`

Tách frame thành:

- root_pos
- root_rot
- joint_dof

Đây là format contract giữa motion file và env.

#### 15.2. `_load_motions`

Gọi `_load_motion_pkl` rồi `_process_data`.

Hãy đọc hai hàm đó liền nhau.

#### 15.3. `_fetch_motion_files`

Nếu `motion_file` là:

- `.pkl`: chỉ một clip
- `.yaml`: nhiều clip + trọng số sample

Ý nghĩa:

- repo hỗ trợ train từ single clip hoặc motion dataset

#### 15.4. `_extract_frame_data`

Đây là điểm chuyển cực quan trọng:

- motion file lưu root rotation dạng expmap
- code convert sang quaternion
- joint dof được convert sang joint rotation quaternion thông qua kinematic model

Nghĩa là `MotionLib` phụ thuộc vào `KinCharModel`.

#### 15.5. `_process_data`

Đây là nơi motion “thô” trở thành motion “dùng được cho RL runtime”.

Nó tính:

- normalized motion weights
- motion lengths
- frame start index của từng motion
- root velocity
- root angular velocity
- dof velocity
- wrap delta cho looping motion

Nếu bạn muốn hiểu tại sao env query được “motion ở thời điểm bất kỳ”, đây là file phải đọc kỹ nhất.

#### 15.6. `calc_motion_phase`

Input:

- motion id
- time

Output:

- phase trong `[0, 1]`

Lưu ý:

- looping motion và non-looping motion xử lý khác nhau

#### 15.7. `calc_motion_frame`

Đây là hàm trọng tâm.

Nó:

1. Tìm hai frame lân cận.
2. Tính hệ số blend.
3. Nội suy root position.
4. Slerp root rotation.
5. Slerp joint rotation.
6. Lấy velocity gần đúng từ frame.
7. Cộng thêm loop offset nếu motion wrap.

Đây là cầu nối trực tiếp giữa “motion dataset rời rạc” và “reference state liên tục theo thời gian”.

### Câu hỏi kiểm tra sau `MotionLib`

1. Tại sao phải nội suy giữa hai frame?
2. Tại sao joint rotation dùng quaternion thay vì giữ nguyên expmap?
3. Motion loop mode ảnh hưởng phase và root offset như thế nào?
4. `MotionLib` có biết gì về Isaac Lab không?

Đáp án câu 4 là không.

---

## Giai đoạn 9: Hiểu kinematic character model

### Bước 16. Đọc `mimickit/anim/kin_char_model.py`

File:

- `mimickit/anim/kin_char_model.py`

Đây là lớp toán học trung gian giữa:

- asset của character
- dof representation
- joint rotation
- body pose

### Điều bạn cần hiểu trong file này

#### 16.1. `JointType`

- `ROOT`
- `HINGE`
- `SPHERICAL`
- `FIXED`

Đây là nền tảng để hiểu mapping giữa dof và rotation.

#### 16.2. `Joint`

Class này định nghĩa:

- joint có bao nhiêu dof
- cắt phần dof nào trong vector tổng
- chuyển dof -> quaternion
- chuyển quaternion -> dof

Đây là nơi “ngữ nghĩa từng khớp” được mã hóa.

#### 16.3. `KinCharModel.init`

Nó giữ:

- `body_names`
- `parent_indices`
- `local_translation`
- `local_rotation`
- `joints`

Đây chính là cây động học của character.

#### 16.4. `dof_to_rot` và `rot_to_dof`

Hai hàm này là trung tâm của mọi chuyển đổi representation.

Bạn sẽ gặp lại chúng ở:

- `CharEnv`
- `DeepMimicEnv`
- `MotionLib`

#### 16.5. `forward_kinematics`

Input:

- root pose
- joint rotations

Output:

- world body positions
- world body rotations

Đây là phép toán cực kỳ quan trọng vì:

- reward thường dùng key body positions
- reset rigid body state cần body pose
- ref motion cần body pose để so sánh

#### 16.6. `compute_dof_vel`

Từ hai frame rotation liên tiếp, nó suy ra vận tốc dof.

Điều này rất quan trọng để build velocity target từ motion clip.

### Sau `kin_char_model.py`, bạn phải hiểu được

1. Vì sao motion file chỉ cần lưu root + joint dof mà vẫn suy ra body positions được.
2. Vì sao DeepMimic reward có thể so sánh key body positions dù motion clip không lưu trực tiếp tất cả body positions.

---

## Giai đoạn 10: Hiểu learning framework

### Bước 17. Đọc `mimickit/learning/base_agent.py`

File:

- `mimickit/learning/base_agent.py`

Đây là file điều phối training loop phía agent.

### Vai trò của `BaseAgent`

1. Quản lý mode train/test.
2. Quản lý normalizer.
3. Quản lý experience buffer.
4. Quản lý return tracking.
5. Điều phối rollout.
6. Điều phối logging.
7. Định nghĩa khung chung cho thuật toán RL.

### Cách đọc file này

#### 17.1. Constructor

Flow:

1. Lưu `env`, `device`, `config`
2. `_load_params`
3. `_build_normalizers`
4. `_build_model`
5. `_build_optimizer`
6. `_build_exp_buffer`
7. `_build_return_tracker`

Điều rất quan trọng:

- Agent nhận `env` ngay trong constructor
- Vì vậy model/action space/obs space đều có thể phụ thuộc env thực tế

#### 17.2. `train_model`

Đây là training loop cấp cao.

Flow:

1. Tạo logger.
2. Reset env lần đầu.
3. Lặp cho tới khi đủ samples:
   - `_train_iter()`
   - cập nhật sample count
   - định kỳ chạy `test_model`
   - log info
   - save model

Bạn nên phân biệt:

- train iteration
- env step
- gradient step
- sample count

Đây là 4 khái niệm khác nhau.

#### 17.3. `_rollout_train`

Flow:

1. Agent quyết định action.
2. Record dữ liệu pre-step.
3. Step env.
4. Update return tracker.
5. Record dữ liệu post-step.
6. Reset những env đã done.
7. Tăng con trỏ experience buffer.

Đây là nơi loop RL thật sự xảy ra.

#### 17.4. `_reset_done_envs`

Điểm đẹp của thiết kế:

- env vectorized
- env nào done thì reset riêng
- không cần chờ cả batch kết thúc cùng lúc

#### 17.5. `_compute_succ_val` và `_compute_fail_val`

Done flag không chỉ để báo hết episode, mà còn ảnh hưởng bootstrap target.

Điều này nối trực tiếp với `DoneFlags` ở `BaseEnv`.

### Câu hỏi kiểm tra sau `BaseAgent`

1. Agent có trực tiếp gọi Isaac Lab API không?
2. Agent giao tiếp với simulator qua lớp nào?
3. `BaseAgent` có biết DeepMimic reward formula không?

Đáp án:

- Không
- Qua `env`
- Không

---

## Giai đoạn 11: Hiểu PPO implementation

### Bước 18. Đọc `mimickit/learning/ppo_agent.py`

File:

- `mimickit/learning/ppo_agent.py`

Đây là implementation học policy cho case DeepMimic humanoid PPO.

### Các phần bạn cần hiểu

#### 18.1. `_load_params`

Nạp các hyperparameter PPO:

- actor/critic epochs
- batch size
- `td_lambda`
- `ppo_clip_ratio`
- `norm_adv_clip`
- regularization weights
- exploration annealing

#### 18.2. `_build_model`

Tạo `PPOModel`.

Bạn đọc file model sau khi đã hiểu `PPOAgent`, không cần đọc model quá sớm.

#### 18.3. `_decide_action`

Flow:

1. Normalize observation.
2. Chạy actor để lấy action distribution.
3. Nếu train:
   - sample action ngẫu nhiên hoặc lấy mode tùy exploration probability
4. Nếu test:
   - lấy mode
5. Unnormalize action về action space thật

Điểm rất quan trọng:

- model làm việc trong normalized action space
- env làm việc trong action thật

#### 18.4. `_build_train_data`

Hàm này dựng target để train.

Nó:

1. Tính next value.
2. Thay value ở trạng thái success/fail bằng terminal value tương ứng.
3. Tính TD-lambda return.
4. Tính advantage.
5. Chuẩn hóa advantage.

Nếu bạn muốn hiểu PPO trong repo này khác bản textbook ở đâu, hãy đọc kỹ hàm này.

#### 18.5. `_update_model`

Tách riêng:

- update critic
- update actor

Mỗi phần có batch size và số epochs riêng.

#### 18.6. `_compute_actor_loss`

Đây là chỗ PPO clip objective được áp dụng.

Ngoài ra còn có thể cộng thêm:

- action bound loss
- entropy loss
- action regularization

### Câu hỏi kiểm tra sau `PPOAgent`

1. Action trước khi vào env đã được unnormalize chưa?
2. Advantage được tính từ reward nào?
3. Agent có biết motion reference không?

Đáp án câu 3:

- agent không trực tiếp biết motion reference
- nó chỉ thấy observation và reward do env cung cấp

---

## Giai đoạn 12: Quay lại Isaac Lab để nối toàn bộ hệ thống

### Bước 19. Đọc `mimickit/engines/isaac_lab_engine.py`

File:

- `mimickit/engines/isaac_lab_engine.py`

Bây giờ mới là lúc đọc file này.

Lúc này bạn đã biết:

- env cần engine làm gì
- reward/obs lấy dữ liệu gì
- reset cần set state gì
- action cần truyền vào simulator theo dạng gì

Nên khi đọc Isaac Lab engine bạn sẽ không bị ngợp vì API.

### Vai trò của `IsaacLabEngine`

Nó là adapter giữa:

- interface `Engine`
- Isaac Lab / Isaac Sim runtime thật

### Cách đọc file này

#### 19.1. Constructor

Điểm cần chú ý:

- tính `self._timestep = 1 / control_freq`
- tính `self._sim_steps = sim_freq / control_freq`
- `sim_timestep = 1 / sim_freq`

Đây là mapping quan trọng giữa:

- tần số physics
- tần số control
- tần số env step

#### 19.2. `create_env`

Mỗi lần env logic gọi `engine.create_env()`, engine chỉ ghi nhận cấu hình cho một env mới.

Việc build simulator object thật sự diễn ra muộn hơn ở `initialize_sim()`.

Đây là điểm kiến trúc cần nhớ:

- `create_env` và `create_obj` chủ yếu đăng ký blueprint
- `initialize_sim` mới materialize scene vào Isaac Lab

#### 19.3. `create_obj`

Nó nhận:

- `obj_type`
- `asset_file`
- `is_visual`
- `enable_self_collisions`
- `fix_root`
- `start_pos`
- `start_rot`
- `color`
- `disable_motors`

Điều này cho thấy `CharEnv` và `DeepMimicEnv` mô tả object ở mức abstraction chung, còn engine quyết định cách hiện thực chúng trong simulator.

#### 19.4. `initialize_sim`

Đây là hàm rất quan trọng.

Nó làm các việc như:

- validate envs
- tạo cloner
- build envs
- build objects
- build contact sensors
- filter collisions
- reset simulator
- build tensors ánh xạ thứ tự body/sensor
- build simulation tensors

Nếu có bug kiểu:

- body order sai
- contact mapping sai
- object clone sai

thì đây là vùng cần kiểm tra đầu tiên.

#### 19.5. `step`

Flow:

1. Update reset objects nếu có.
2. Nếu đang record video thì capture frame.
3. Lặp `self._sim_steps` lần:
   - pre sim step
   - sim step
   - post sim step
4. Clear forces

Đây là cách engine đảm bảo:

- env step ở `control_freq`
- physics chạy ở `sim_freq`

#### 19.6. `set_cmd`

Action từ policy đi vào đây.

Engine sẽ:

1. đổi thứ tự dof từ common order sang simulator order
2. tùy `control_mode` mà gọi:
   - position target
   - velocity target
   - effort target

Đây là đoạn cực kỳ quan trọng nếu:

- action chạy sai khớp
- character rung bất thường
- policy output có vẻ đúng nhưng simulator phản ứng sai

#### 19.7. Các getter như `get_root_pos`, `get_dof_pos`, `get_body_pos`

Đây là cầu nối để env lấy state từ simulator.

Khi đọc các getter, hãy luôn hỏi:

1. Dữ liệu đang ở world frame hay local frame?
2. Có cần đổi thứ tự quaternion không?
3. Có cần trừ `env_offset` không?
4. Có cần đổi thứ tự dof/body về “common order” không?

Ví dụ:

- code có xử lý đổi quaternion giữa `wxyz` và `xyzw`
- code có trừ `env_offsets` để trả nhân vật về frame riêng của env

Đây là chi tiết rất dễ bug nếu bạn tự sửa engine.

### Khi đọc `IsaacLabEngine`, bạn không cần hiểu hết mọi API Isaac Lab

Bạn chỉ cần hiểu:

1. Nó đang implement method nào của `Engine`.
2. Method đó phục vụ nhu cầu nào của `env`.
3. Nó có thực hiện chuyển đổi dữ liệu nào trước khi trả về không.

---

## Giai đoạn 13: Ghép tất cả thành một luồng hoàn chỉnh

Đây là luồng chạy đầy đủ cho case:

- `DeepMimic`
- `humanoid`
- `PPO`
- `Isaac Lab`

### Lúc khởi động

1. Bạn chạy `python mimickit/run.py ...`
2. `run.py` load args và YAML
3. `env_builder` tạo `DeepMimicEnv`
4. `DeepMimicEnv` qua `SimEnv` build `IsaacLabEngine`
5. `CharEnv` build character từ `char_file`
6. `DeepMimicEnv` load motion từ `motion_file`
7. `agent_builder` tạo `PPOAgent`

### Lúc reset env

1. `DeepMimicEnv` sample motion id và thời điểm motion
2. `MotionLib` nội suy frame motion
3. Env ghi kết quả vào `_ref_*`
4. Env set state của character thật bằng reference state
5. Observation ban đầu được tính ra

### Mỗi env step

1. `PPOAgent` lấy obs hiện tại
2. Actor sinh action
3. `CharEnv._apply_action` gọi `engine.set_cmd`
4. `IsaacLabEngine.step` chạy nhiều physics substeps
5. `DeepMimicEnv._update_ref_motion` cập nhật target motion theo thời gian mới
6. `DeepMimicEnv._compute_obs` dựng obs mới
7. `DeepMimicEnv._update_reward` so sánh character thật với reference
8. `DeepMimicEnv._update_done` xét timeout/fail
9. `BaseAgent` ghi dữ liệu vào buffer

### Mỗi training iteration

1. Agent rollout `steps_per_iter` bước
2. PPO tính return/advantage
3. Critic update
4. Actor update
5. Định kỳ test model và log

Nếu bạn mô tả trôi chảy được chuỗi trên mà không cần nhìn file, tức là bạn đã hiểu kiến trúc cốt lõi của repo.

---

## Cách tự học repo trong 7 buổi

Nếu muốn học có hệ thống, bạn có thể chia như sau.

### Buổi 1: Tổng quan và config

Đọc:

- `README.md`
- `data/engines/isaac_lab_engine.yaml`
- `data/envs/deepmimic_humanoid_env.yaml`
- `data/agents/deepmimic_humanoid_ppo_agent.yaml`

Kết quả mong muốn:

- hiểu repo chạy case nào
- biết ba tầng config

### Buổi 2: Luồng dựng object

Đọc:

- `mimickit/run.py`
- `mimickit/envs/env_builder.py`
- `mimickit/learning/agent_builder.py`
- `mimickit/engines/engine_builder.py`

Kết quả mong muốn:

- vẽ được luồng `args -> env/agent/engine`

### Buổi 3: Simulator abstraction

Đọc:

- `mimickit/engines/engine.py`
- `mimickit/envs/base_env.py`
- `mimickit/envs/sim_env.py`

Kết quả mong muốn:

- hiểu khung step/reset/render/buffer

### Buổi 4: Character layer

Đọc:

- `mimickit/envs/char_env.py`
- character asset tương ứng trong `data/assets/`

Kết quả mong muốn:

- hiểu action space, observation cơ bản, reset character

### Buổi 5: Motion imitation layer

Đọc:

- `mimickit/envs/deepmimic_env.py`
- `mimickit/anim/motion.py`
- `mimickit/anim/motion_lib.py`
- `mimickit/anim/kin_char_model.py`

Kết quả mong muốn:

- hiểu reference motion, reward, done, target obs

### Buổi 6: RL layer

Đọc:

- `mimickit/learning/base_agent.py`
- `mimickit/learning/ppo_agent.py`
- `mimickit/learning/ppo_model.py`

Kết quả mong muốn:

- hiểu rollout, return, PPO update

### Buổi 7: Isaac Lab adapter

Đọc:

- `mimickit/engines/isaac_lab_engine.py`

Kết quả mong muốn:

- hiểu env logic được map vào simulator như thế nào

---

## Checklist “đã hiểu đến đâu”

### Mức 1: Biết chạy

Bạn biết:

- config nào dùng cho Isaac Lab
- config nào dùng cho DeepMimic humanoid
- train/test bằng lệnh nào

### Mức 2: Biết luồng code

Bạn biết:

- `run.py` gọi builder nào
- builder nào tạo env/agent/engine nào

### Mức 3: Biết environment

Bạn biết:

- obs đến từ đâu
- reward đến từ đâu
- done đến từ đâu
- reset lấy motion reference thế nào

### Mức 4: Biết simulator integration

Bạn biết:

- action vào Isaac Lab ở đâu
- state từ Isaac Lab ra ở đâu
- dof/quaternion/body order được map ở đâu

### Mức 5: Biết sửa code có chủ đích

Bạn có thể tự tin sửa:

- reward weights
- observation features
- termination condition
- action control mode
- motion dataset config

---

## Các câu hỏi rất nên tự trả lời khi đọc

1. Repo này simulator-agnostic ở mức nào, và phụ thuộc backend ở mức nào?
2. Phần nào là “task logic”, phần nào là “physics backend”?
3. Observation nào là Markov-ish cho imitation task?
4. Vì sao cần target future observation?
5. Vì sao reset theo random frame của motion thường giúp học tốt hơn?
6. Vì sao body names giữa kinematic model và simulator model phải khớp?
7. Tại sao reward tracking pose thường chưa đủ, mà còn cần root vel và key body positions?
8. Nếu action control mode đổi từ `pos` sang `torque`, phần nào của hệ thống phải đổi cách hiểu?

Nếu bạn trả lời mượt 8 câu này, bạn đã nắm repo khá chắc.

---

## Những điểm dễ nhầm khi học repo này

### Nhầm 1. Tưởng Isaac Lab là trung tâm của repo

Sai.

Isaac Lab chỉ là một engine backend.
Trung tâm của repo là abstraction giữa:

- motion
- env logic
- RL agent
- engine backend

### Nhầm 2. Tưởng `DeepMimicEnv` tự làm physics

Sai.

`DeepMimicEnv` chỉ định nghĩa logic task.
Physics thật nằm ở engine.

### Nhầm 3. Tưởng motion file lưu mọi thứ cần cho reward

Sai.

Motion file chỉ lưu pose theo representation gọn.
Body positions và nhiều đại lượng khác được suy ra qua kinematic model.

### Nhầm 4. Tưởng PPO agent biết đang imitation motion nào

Sai theo nghĩa trực tiếp.

Agent không biết “motion file” như object logic.
Nó chỉ nhận observation và reward do env cung cấp.

### Nhầm 5. Tưởng reset chỉ là set joint angle

Sai.

Reset còn phải đồng bộ root pose, dof vel, body pose, rigid body state.

---

## Nếu muốn đào sâu hơn sau khi hiểu trục chính

Sau khi nắm `DeepMimic + PPO + Isaac Lab`, bạn có thể mở rộng theo 3 hướng.

### Hướng 1: Học các env/task khác

Đọc thêm:

- `mimickit/envs/amp_env.py`
- `mimickit/envs/ase_env.py`
- `mimickit/envs/add_env.py`
- `mimickit/envs/smp_env.py`
- `mimickit/envs/task_location_env.py`
- `mimickit/envs/task_steering_env.py`
- `mimickit/envs/task_dodgeball_env.py`

Mục tiêu:

- xem framework tái sử dụng `CharEnv` và `SimEnv` ra sao

### Hướng 2: Học các learning algorithm khác

Đọc thêm:

- `amp_agent.py`, `amp_model.py`
- `awr_agent.py`, `awr_model.py`
- `ase_agent.py`, `ase_model.py`
- `add_agent.py`, `add_model.py`
- `lcp_agent.py`, `lcp_model.py`

Mục tiêu:

- hiểu repo mở rộng từ PPO nền như thế nào

### Hướng 3: Học backend abstraction sâu hơn

So sánh:

- `isaac_gym_engine.py`
- `isaac_lab_engine.py`
- `newton_engine.py`

Mục tiêu:

- hiểu phần nào là interface chung
- phần nào là đặc thù backend

---

## Cách đọc source hiệu quả nhất cho repo này

Đề xuất cách làm thực tế:

1. Mỗi buổi chỉ theo một luồng duy nhất.
2. Vừa đọc vừa vẽ sơ đồ dữ liệu đi qua các class.
3. Với mỗi file, chỉ trả lời 3 câu:
   - file này chịu trách nhiệm gì
   - input của nó đến từ đâu
   - output của nó đi đâu
4. Với mỗi tensor quan trọng, ghi lại:
   - shape
   - semantic
   - frame tọa độ
5. Sau mỗi bước, quay lại case `DeepMimic humanoid + Isaac Lab` để nối vào luồng tổng.

Nếu muốn học rất chắc, bạn có thể tạo bảng 4 cột:

- File
- Class/Hàm quan trọng
- Input
- Output

Repo này rất hợp với cách học đó.

---

## Kết luận ngắn

Nếu chỉ nhớ 5 điểm cốt lõi, hãy nhớ 5 điểm này:

1. `run.py` chỉ điều phối, không chứa task logic.
2. `envs/` mới là nơi định nghĩa imitation task.
3. `anim/` là nền tảng để biến motion clip thành reference state liên tục.
4. `learning/` chỉ học từ observation/reward mà env sinh ra.
5. `IsaacLabEngine` là lớp adapter nối abstraction chung vào Isaac Lab.

---

## Thứ tự file nên mở ngay bây giờ

Nếu bạn muốn bắt đầu ngay theo đúng trình tự tối ưu, hãy mở theo thứ tự này:

1. `README.md`
2. `data/engines/isaac_lab_engine.yaml`
3. `data/envs/deepmimic_humanoid_env.yaml`
4. `data/agents/deepmimic_humanoid_ppo_agent.yaml`
5. `mimickit/run.py`
6. `mimickit/envs/env_builder.py`
7. `mimickit/learning/agent_builder.py`
8. `mimickit/engines/engine_builder.py`
9. `mimickit/engines/engine.py`
10. `mimickit/envs/base_env.py`
11. `mimickit/envs/sim_env.py`
12. `mimickit/envs/char_env.py`
13. `mimickit/envs/deepmimic_env.py`
14. `mimickit/anim/motion.py`
15. `mimickit/anim/motion_lib.py`
16. `mimickit/anim/kin_char_model.py`
17. `mimickit/learning/base_agent.py`
18. `mimickit/learning/ppo_agent.py`
19. `mimickit/learning/ppo_model.py`
20. `mimickit/engines/isaac_lab_engine.py`

Nếu bạn muốn, bước tiếp theo mình có thể làm tiếp cho bạn một tài liệu số 2:

- bản “reading worksheet” dạng checklist từng file
- hoặc bản “call graph” mô tả luồng hàm từ `run.py` đến `IsaacLabEngine.step()`
- hoặc bản “giải thích từng tensor quan trọng” trong `DeepMimicEnv`
