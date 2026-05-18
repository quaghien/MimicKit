# `env_builder.build_env` chi tiết

## Case cụ thể đang phân tích

Case này cố định là:

- Engine: `Isaac Lab`
- Environment: `DeepMimic`
- Character: `humanoid`
- Agent: `PPO`

Các file config tương ứng:

- Engine config: [`data/engines/isaac_lab_engine.yaml`](/home/wanhin/hienhq/MimicKit/data/engines/isaac_lab_engine.yaml)
- Env config: [`data/envs/deepmimic_humanoid_env.yaml`](/home/wanhin/hienhq/MimicKit/data/envs/deepmimic_humanoid_env.yaml)
- Agent config: [`data/agents/deepmimic_humanoid_ppo_agent.yaml`](/home/wanhin/hienhq/MimicKit/data/agents/deepmimic_humanoid_ppo_agent.yaml)

Lưu ý:

- preset [`args/deepmimic_humanoid_ppo_args.txt`](/home/wanhin/hienhq/MimicKit/args/deepmimic_humanoid_ppo_args.txt) mặc định đang trỏ sang `isaac_gym_engine.yaml`, không phải `isaac_lab_engine.yaml`
- nếu muốn đúng case này thì phải override `--engine_config`

Ví dụ lệnh đúng cho case đang phân tích:

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --mode train
```

---

## 1. Flow tổng từ lệnh chạy đến `env_builder.build_env`

Trong [`mimickit/run.py`](/home/wanhin/hienhq/MimicKit/mimickit/run.py):

- `main()` ở dòng `134`
- `run()` ở dòng `97`
- `build_env()` ở dòng `33`

Flow thật:

```txt
main()
  -> load_args()
  -> run(...)
       -> build_env(args, num_envs, device, visualize)
            -> env_builder.build_env(env_file, engine_file, ...)
```

### Ý nghĩa từng chỗ

- `run.py:22-31` `load_args(argv)`: nạp CLI và `arg_file`
- `run.py:97-115` `run(...)`: parse mode, num_envs, visualize, rồi gọi `build_env(...)`
- `run.py:33-39` `build_env(...)`: chỉ lấy ra `env_config`, `engine_config`, `video`, rồi chuyển hết cho `env_builder.build_env(...)`

Nói ngắn:

- `run.py` không tự build chi tiết env
- `run.py` chỉ chuyển đường dẫn config xuống builder layer

---

## 2. `env_builder.build_env(...)` làm gì

File: [`mimickit/envs/env_builder.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/env_builder.py)

### 2.1. Entry function

- `build_env(...)` ở dòng `8`

Code logic chính:

1. `load_configs(env_file, engine_file)` ở dòng `9`
2. lấy `env_name = env_config["env_name"]` ở dòng `11`
3. rẽ nhánh theo `env_name`
4. với case này, `env_name == "deepmimic"` nên đi vào dòng `17-19`
5. tạo `deepmimic_env.DeepMimicEnv(...)`

### 2.2. Ý nghĩa thật

`env_builder.build_env(...)` là chỗ quyết định:

- bài toán nào sẽ được dựng
- class env nào sẽ chịu trách nhiệm logic RL

Với config hiện tại:

- `env_name` trong env config là `"deepmimic"` tại [`data/envs/deepmimic_humanoid_env.yaml:1`](/home/wanhin/hienhq/MimicKit/data/envs/deepmimic_humanoid_env.yaml:1)
- nên builder tạo `DeepMimicEnv`

### 2.3. Tại sao import `engine_builder` ở đầu file

- `env_builder.py:3-4`

Import này không trực tiếp gọi engine ngay trong file này.

Ý nghĩa:

- đảm bảo import order đúng cho một số simulator backend
- engine thật sẽ được gọi sâu hơn trong constructor chain của `SimEnv`

---

## 3. `load_configs(...)` làm gì

Trong [`env_builder.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/env_builder.py):

- `load_config(file)` ở dòng `55`
- `load_configs(env_file, engine_file)` ở dòng `63`
- `override_engine_config(...)` ở dòng `73`

### 3.1. `load_config(file)` dòng `55-61`

Ý nghĩa:

- đọc YAML bằng `yaml.safe_load`
- trả về Python dict

Với case này:

- `env_file = data/envs/deepmimic_humanoid_env.yaml`
- `engine_file = data/engines/isaac_lab_engine.yaml`

### 3.2. `load_configs(...)` dòng `63-71`

Ý nghĩa:

- load cả `env_config` và `engine_config`
- nếu trong `env_config` có block `engine`, thì env được quyền override một phần engine config

Case `deepmimic_humanoid_env.yaml` hiện tại:

- không có block `engine`
- nên `engine_config` giữ nguyên đúng như file `isaac_lab_engine.yaml`

### 3.3. `override_engine_config(...)` dòng `73-85`

Ý nghĩa:

- merge key từ `env_config["engine"]` đè lên `engine_config`
- giúp một env cụ thể điều chỉnh backend mà không cần sửa file engine gốc

Case này:

- hàm tồn tại nhưng không được dùng

---

## 4. Config cụ thể đang chảy qua `env_builder.build_env`

## 4.1. Engine config

File: [`data/engines/isaac_lab_engine.yaml`](/home/wanhin/hienhq/MimicKit/data/engines/isaac_lab_engine.yaml)

```yaml
engine_name: "isaac_lab"
control_mode: "pos"
control_freq: 30
sim_freq: 120
env_spacing: 5
```

Ý nghĩa từng key:

- `engine_name = "isaac_lab"`: `engine_builder` sẽ tạo `IsaacLabEngine`
- `control_mode = "pos"`: action của policy được hiểu là joint position target
- `control_freq = 30`: policy đổi lệnh 30 lần mỗi giây
- `sim_freq = 120`: physics chạy 120 bước mỗi giây
- `env_spacing = 5`: khi clone nhiều env, mỗi env đặt cách nhau 5m theo offset layout

Ý nghĩa logic:

- một lần control kéo dài `1/30s`
- một physics step kéo dài `1/120s`
- nên một action sẽ được giữ trong `4` physics steps

## 4.2. Env config

File: [`data/envs/deepmimic_humanoid_env.yaml`](/home/wanhin/hienhq/MimicKit/data/envs/deepmimic_humanoid_env.yaml)

### Nhóm nhận diện bài toán

- `env_name = "deepmimic"`: builder sẽ tạo `DeepMimicEnv`
- `char_file = "data/assets/humanoid/humanoid.xml"`: dùng humanoid MJCF làm character
- `motion_file = "data/motions/humanoid/humanoid_spinkick.pkl"`: motion reference là clip spinkick

### Nhóm simulation/task cơ bản

- `camera_mode = "track"`: camera bám nhân vật khi visualize
- `episode_length = 10.0`: episode tối đa 10 giây
- `enable_early_termination = True`: được terminate sớm nếu fail

### Nhóm observation

- `global_obs = True`: observation ở hệ global
- `root_height_obs = True`: giữ thông tin chiều cao root
- `enable_phase_obs = False`: không thêm phase obs
- `enable_tar_obs = True`: có thêm future target observation
- `num_phase_encoding = 4`: số chiều encoding phase nếu bật phase obs
- `tar_obs_steps = [1, 2, 3]`: lấy target tại 3 step tương lai

### Nhóm reset/reference

- `rand_reset = True`: reset ngẫu nhiên tại các thời điểm khác nhau trong motion
- `ref_char_offset = [2.0, 0.0, 0.0]`: ref character đứng lệch sang bên 2m
- `init_pose = [...]`: tư thế khởi tạo mặc định

### Nhóm termination/contact

- `pose_termination = True`: terminate nếu pose lệch quá xa
- `pose_termination_dist = 1.0`: ngưỡng lệch 1m
- `contact_bodies = ["right_foot", "left_foot"]`: chỉ 2 chân được coi là contact hợp lệ

### Nhóm reward

- `key_bodies = ["head", "right_hand", "left_hand", "right_foot", "left_foot"]`
- `joint_err_w = [...]`
- `reward_pose_w = 0.5`
- `reward_vel_w = 0.1`
- `reward_root_pose_w = 0.15`
- `reward_root_vel_w = 0.1`
- `reward_key_pos_w = 0.15`
- `reward_pose_scale = 0.25`
- `reward_vel_scale = 0.01`
- `reward_root_pose_scale = 5.0`
- `reward_root_vel_scale = 1.0`
- `reward_key_pos_scale = 10.0`

Ý nghĩa logic:

- reward tổng là tổ hợp của pose, velocity, root pose, root velocity, key body positions
- từng thành phần reward so actor state với reference motion state

## 4.3. Agent config

File: [`data/agents/deepmimic_humanoid_ppo_agent.yaml`](/home/wanhin/hienhq/MimicKit/data/agents/deepmimic_humanoid_ppo_agent.yaml)

File này không đi vào `env_builder.build_env`, nhưng nó xác định agent sẽ dùng env vừa build như thế nào.

Với case này:

- `agent_name = "PPO"`
- `steps_per_iter = 32`
- `actor_epochs = 5`
- `critic_epochs = 2`

Ý nghĩa trong flow:

- env được build xong trước
- sau đó PPOAgent cầm env này để rollout 32 step mỗi iteration

---

## 5. Constructor chain thật sau khi `env_builder` chọn `DeepMimicEnv`

Sau `env_builder.py:17-19`, object được tạo theo chuỗi:

```txt
DeepMimicEnv.__init__
  -> CharEnv.__init__
      -> SimEnv.__init__
          -> BaseEnv.__init__
          -> _build_engine(...)
          -> _build_envs(...)
          -> engine.initialize_sim()
          -> _build_action_space()
          -> _build_sim_tensors()
          -> _build_data_buffers()
```

Đây là phần quan trọng nhất của cả flow build env.

---

## 6. `DeepMimicEnv.__init__` nhận config nào và làm gì

File: [`mimickit/envs/deepmimic_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/deepmimic_env.py)

- class `DeepMimicEnv` ở dòng `12`
- `__init__` ở dòng `13-45`

### 6.1. Nó đọc key nào từ env config

`DeepMimicEnv.__init__` đọc và cache các key sau:

- `enable_early_termination` dòng `14`
- `num_phase_encoding` dòng `15`
- `pose_termination` dòng `17`
- `pose_termination_dist` dòng `18`
- `enable_phase_obs` dòng `19`
- `enable_tar_obs` dòng `20`
- `tar_obs_steps` dòng `21-22`
- `rand_reset` dòng `23`
- `ref_char_offset` dòng `25`
- `log_tracking_error` dòng `26`
- toàn bộ trọng số reward ở dòng `28-38`
- `visualize_ref_char` dòng `40`

### 6.2. Ý nghĩa logic

`DeepMimicEnv` là lớp thêm logic imitation vào trên nền `CharEnv`.

Nó chịu trách nhiệm cho:

- motion reference
- reset theo motion
- target observation
- reward tracking motion
- done theo pose/contact/motion end
- ref character để visualize

### 6.3. Sau khi đọc config thì làm gì

- dòng `42-44` gọi `super().__init__(...)`

Nghĩa là:

- `DeepMimicEnv` đọc các tham số imitation trước
- sau đó giao phần dựng simulator/env/object cho chain bên dưới

---

## 7. `CharEnv.__init__` làm gì

File: [`mimickit/envs/char_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/char_env.py)

- class `CharEnv` ở dòng `17`
- `__init__` ở dòng `18-30`

### 7.1. Nó đọc key nào

- `global_obs` dòng `19`
- `root_height_obs` dòng `20`
- `zero_center_action` dòng `21`

### 7.2. Ý nghĩa logic

`CharEnv` là lớp trung gian cho các bài toán có character articulated.

Nó chịu trách nhiệm:

- load kinematic character model từ `char_file`
- parse `init_pose`
- build action space theo giới hạn joint
- map body names sang body ids
- reset character state
- convert state simulator thành obs dạng character

### 7.3. Sau super thì làm thêm gì

- dòng `27-29`
- in thuộc tính character
- validate body order giữa simulator model và kinematic model

Nghĩa là:

- sau khi env được dựng xong, `CharEnv` kiểm tra tính khớp cấu trúc mô hình

---

## 8. `SimEnv.__init__` là lõi của flow build

File: [`mimickit/envs/sim_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/sim_env.py)

- class `SimEnv` ở dòng `18`
- `__init__` ở dòng `21-40`

Đây là hàm quan trọng nhất nếu bạn muốn hiểu `env_builder.build_env` thật sự dẫn tới đâu.

### 8.1. Thứ tự chạy thật trong `SimEnv.__init__`

1. `BaseEnv.__init__(visualize)` dòng `22`
2. lưu `device` dòng `24`
3. đọc `episode_length` dòng `25`
4. `self._build_engine(...)` dòng `27`
5. `self._build_envs(env_config, num_envs)` dòng `28`
6. `self._engine.initialize_sim()` dòng `29`
7. `self._build_action_space()` dòng `31`
8. `self._build_sim_tensors(env_config)` dòng `32`
9. `self._build_data_buffers()` dòng `33`
10. nếu visualize thì build camera và GUI dòng `35-38`

### 8.2. Ý nghĩa cực quan trọng

Thứ tự này nói lên kiến trúc thật:

- build engine trước
- đăng ký env/object với engine trước
- engine mới initialize simulator thật
- sau khi simulator có object thật rồi, env mới biết action bounds, tensor shapes, observation shapes

Nghĩa là:

- action space không thể build trước khi simulator/object tồn tại
- sim tensors cũng không thể build trước khi `initialize_sim()` xong

---

## 9. `SimEnv._build_engine(...)` sẽ đi đâu với case `isaac_lab`

Trong [`sim_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/sim_env.py):

- `_build_engine(...)` ở dòng `180-182`

Nó gọi:

- [`mimickit/engines/engine_builder.py`](/home/wanhin/hienhq/MimicKit/mimickit/engines/engine_builder.py) dòng `6-21`

### Flow cụ thể

- `config["engine_name"]` ở `engine_builder.py:7`
- vì bằng `"isaac_lab"` nên đi vào `engine_builder.py:12-14`
- tạo `IsaacLabEngine(config, num_envs, device, visualize, record_video=...)`

---

## 10. `IsaacLabEngine.__init__` làm gì với engine config

File: [`mimickit/engines/isaac_lab_engine.py`](/home/wanhin/hienhq/MimicKit/mimickit/engines/isaac_lab_engine.py)

- class `IsaacLabEngine` ở dòng `62`
- `__init__` ở dòng `63-105`

### 10.1. Key nào được đọc

- `sim_freq` dòng `67`
- `control_freq` dòng `68`
- `env_spacing` dòng `81`
- `control_mode` dòng `85-88`

### 10.2. Ý nghĩa thật của từng key

- `sim_freq = 120`
  physics dt = `1 / 120`
- `control_freq = 30`
  env/agent timestep = `1 / 30`
- dòng `69-70`
  assert để đảm bảo `sim_freq` là bội số của `control_freq`
- dòng `72`
  `self._timestep = 1/control_freq`
- dòng `73`
  `self._sim_steps = sim_freq/control_freq = 4`

Nghĩa là:

- mỗi lần env gọi `engine.step()`
- Isaac Lab engine sẽ chạy `4` physics substeps

### 10.3. Còn `control_mode = "pos"` thì sao

- dòng `85-88`
- map string `"pos"` thành `engine.ControlMode.pos`

Ý nghĩa:

- về sau khi env gọi `engine.set_cmd(...)`
- action sẽ được áp qua `set_joint_position_target(...)`
- xem `isaac_lab_engine.py:181-198`

### 10.4. `env_spacing = 5` để làm gì

- dòng `90` tính `self._env_offsets = self._compute_env_offsets(num_envs)`
- khi tạo object, `create_obj(...)` ở dòng `150-179` sẽ cộng offset env vào `start_pos`

Ý nghĩa:

- mỗi env clone sống ở một vị trí khác nhau trong world
- tránh chồng chéo hình học khi clone nhiều humanoid

---

## 11. `self._build_envs(env_config, num_envs)` thật sự làm gì

Call này xảy ra ở `sim_env.py:28`.

Vì object hiện tại là `DeepMimicEnv`, Python sẽ dispatch xuống:

- `DeepMimicEnv._build_envs(...)` dòng `295-302`

Flow:

1. tạo `self._ref_char_ids = []` dòng `296`
2. gọi `super()._build_envs(...)` dòng `298`
3. sau khi build xong env/object, load motion file ở dòng `300-301`

### Ý nghĩa logic

- build tất cả env và character trước
- motion library load sau
- vì motion cần kinematic character model đã sẵn sàng

---

## 12. `CharEnv._build_envs(...)` làm gì

File: [`char_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/char_env.py)

- `_build_envs(...)` ở dòng `52-68`

### Flow thật

1. đọc `char_file` dòng `53`
2. `_build_kin_char_model(char_file)` dòng `54`
3. đọc `init_pose` dòng `56`
4. `_parse_init_pose(...)` dòng `57`
5. tạo list `self._char_ids` dòng `59`
6. loop `for e in range(num_envs)` dòng `61`
7. `env_id = self._engine.create_env()` dòng `63`
8. `self._build_env(env_id, env_config)` dòng `65`

### Ý nghĩa logic

Trước khi simulator thật initialize:

- env chỉ đang “đăng ký blueprint”
- mỗi env được đánh số
- mỗi object được báo cho engine biết sẽ cần tạo gì

Đây chưa phải lúc physics đã chạy.

---

## 13. `_build_kin_char_model(char_file)` đọc gì từ humanoid config

Trong [`char_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/char_env.py):

- `_build_kin_char_model(...)` ở dòng `93-110`

Vì `char_file = data/assets/humanoid/humanoid.xml`:

- phần mở rộng là `.xml`
- nên đi vào dòng `95-97`
- tạo `MJCFCharModel`
- rồi `load(char_file)` ở dòng `109`

### Ý nghĩa logic

Kinematic model dùng để:

- biết joint nào tồn tại
- body nào tồn tại
- mỗi joint có mấy dof
- đổi qua lại giữa `dof_pos` và `joint_rot`
- forward kinematics để tính vị trí/rotation các body

Nếu không có bước này thì DeepMimic không thể:

- so sánh pose với motion reference
- build action bounds theo joint structure
- build key body ids theo tên

---

## 14. `_parse_init_pose(...)` biến `init_pose` thành gì

Trong [`char_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/char_env.py):

- `_parse_init_pose(...)` ở dòng `32-50`

Ý nghĩa:

- tách vector `init_pose` thành:
  - `init_root_pos`
  - `init_root_rot`
  - `init_dof_pos`
- convert root rotation từ exponential map sang quaternion ở dòng `48`

### Tại sao cần

`init_pose` trong YAML chỉ là dữ liệu tĩnh.

Simulator lại cần:

- root position
- root rotation quaternion
- dof positions

Nên phải chuyển format trước khi tạo object hoặc reset object.

---

## 15. `_build_env(env_id, env_config)` cụ thể tạo humanoid thế nào

### 15.1. Ở `CharEnv`

- `_build_env(...)` dòng `70-80`

Nó gọi:

- `_build_character(...)` dòng `82-91`

Trong `_build_character(...)`:

- lấy `char_file` dòng `83`
- gọi `self._engine.create_obj(...)` dòng `84-90`
- `obj_type = articulated`
- `asset_file = humanoid.xml`
- `name = "character"`
- `start_pos = self._init_root_pos`
- `start_rot = self._init_root_rot`

Ý nghĩa:

- mỗi env được gắn một humanoid articulated object

### 15.2. Ở `DeepMimicEnv`

- `_build_env(...)` dòng `304-316`

Nó gọi `super()._build_env(...)` trước, tức build humanoid chính trước.

Sau đó:

- nếu visualize ref char được bật
- gọi `_build_ref_character(...)` dòng `318-328`

`_build_ref_character(...)` tạo thêm:

- một object `"ref_character"`
- `is_visual=True`
- `disable_motors=True`
- `enable_self_collisions=False`

Ý nghĩa:

- đây là character tham chiếu để xem motion target
- nó không phải nhân vật policy điều khiển

---

## 16. `IsaacLabEngine.create_env()` và `create_obj()` thật ra mới chỉ đăng ký blueprint

Trong [`isaac_lab_engine.py`](/home/wanhin/hienhq/MimicKit/mimickit/engines/isaac_lab_engine.py):

- `create_env()` dòng `110-115`
- `create_obj()` dòng `150-179`

### `create_env()`

- tạo `env_id` mới
- append một list object config rỗng vào `self._obj_cfgs`

Ý nghĩa:

- engine ghi nhớ env này sẽ chứa những object gì

### `create_obj()`

- chuẩn hóa `start_rot` về format Isaac Lab cần
- cộng `env_offset` vào `start_pos` dòng `163-164`
- đóng gói mọi thứ vào `ObjCfg`
- append `ObjCfg` vào `self._obj_cfgs[env_id]`

Ý nghĩa:

- chưa tạo object vật lý ngay
- mới chỉ lưu “env này cần một humanoid character ở pose này”

Đây là điểm rất quan trọng khi đọc flow:

- `create_obj` chưa phải simulator spawn ngay
- spawn thật diễn ra ở `initialize_sim()`

---

## 17. `engine.initialize_sim()` là lúc simulator thật sự được dựng

Call này xảy ra tại `sim_env.py:29`.

Trong [`isaac_lab_engine.py`](/home/wanhin/hienhq/MimicKit/mimickit/engines/isaac_lab_engine.py):

- `initialize_sim()` ở dòng `117-134`

### Flow thật

1. `_validate_envs()` dòng `120`
2. tạo `Cloner` dòng `121`
3. `_build_envs()` dòng `123`
4. `_build_objs()` dòng `124`
5. `_build_ground_contact_sensors()` dòng `125`
6. `_filter_env_collisions()` dòng `126`
7. `self._sim.reset()` dòng `128-129`
8. build body order tensors dòng `131`
9. build sensor order tensors dòng `132`
10. build sim tensors dòng `133`

### Ý nghĩa logic

Đây là thời điểm:

- stage USD/Isaac Lab được dựng thật
- articulated objects được materialize thật
- sensors được attach
- tensor view qua GPU được xây

Sau bước này env mới có thể:

- đọc root pos
- đọc dof pos
- đọc body positions
- set action target

---

## 18. Sau `initialize_sim()` vì sao mới build action space

Quay lại [`sim_env.py:31`](/home/wanhin/hienhq/MimicKit/mimickit/envs/sim_env.py:31):

- `self._build_action_space()`

Với case này, method thực thi là:

- `CharEnv._build_action_space()` dòng `122-153`

### Flow

1. hỏi `control_mode = self._engine.get_control_mode()` dòng `123`
2. vì case này là `pos`, đi vào dòng `136-140`
3. lấy dof limits từ engine:
   `self._engine.get_obj_dof_limits(0, char_id)`
4. gọi `_build_action_bounds_pos(...)`
5. tạo `spaces.Box(low=low, high=high)` dòng `152`

### Ý nghĩa logic

Action space phụ thuộc vào:

- engine control mode
- số dof thật của humanoid
- joint limits thật trong asset

Nên phải đợi:

- engine dựng xong object
- engine biết dof limits

rồi mới build action space chính xác được.

---

## 19. Sau đó `_build_sim_tensors(env_config)` đi đâu

Call bắt đầu ở `sim_env.py:32`.

Dispatch chain:

- `DeepMimicEnv._build_sim_tensors(...)` dòng `77-109`
- gọi `super()` xuống `CharEnv._build_sim_tensors(...)` dòng `112-120`
- rồi xuống `SimEnv._build_sim_tensors(...)` dòng `188-191`

### 19.1. `SimEnv._build_sim_tensors(...)`

- tạo `self._env_ids = [0, 1, ..., num_envs-1]`

### 19.2. `CharEnv._build_sim_tensors(...)`

- tạo `self._action_bound_low`
- tạo `self._action_bound_high`
- đọc `key_bodies` từ config
- map sang `self._key_body_ids`

### 19.3. `DeepMimicEnv._build_sim_tensors(...)`

- tạo buffer `motion_ids`, `motion_time_offsets`
- lấy từ engine các tensor state:
  - root pos/rot/vel
  - body pos/rot
  - dof pos/vel
- tạo ref tensors tương ứng
- đọc `contact_bodies`
- map sang `self._contact_body_ids`
- đọc `joint_err_w`
- parse thành `self._joint_err_w` và `self._dof_err_w`

### Ý nghĩa logic

Đây là bước “nối config với tensor runtime”.

Sau bước này env đã sẵn sàng để:

- reset theo motion
- tính reward
- tính done
- build observation

---

## 20. `_build_data_buffers()` chuẩn bị buffer runtime gì

Trong chain:

- `SimEnv._build_data_buffers()` dòng `193-206`
- `DeepMimicEnv._build_data_buffers()` dòng `287-293`

### `SimEnv`

Tạo:

- `_reward_buf`
- `_done_buf`
- `_timestep_buf`
- `_time_buf`
- `_obs_buf`
- `_info`

### `DeepMimicEnv`

Nếu `log_tracking_error = True`:

- tạo thêm `_error_tracker`

### Ý nghĩa

Đây là các buffer chính mà agent sẽ đọc ở mỗi bước rollout.

---

## 21. Sau khi build xong, `reset()` sẽ đi flow nào

Khi agent bắt đầu train, env sẽ được reset.

Flow chính:

- `SimEnv.reset()` dòng `64-75`
- `self._reset_envs(reset_env_ids)` dòng `70`

Dispatch chain cho case này:

- `CharEnv._reset_envs(...)` dòng `291-297`
- trong đó gọi `self._reset_char(env_ids)` dòng `295`
- nhưng vì object là `DeepMimicEnv`, method thực thi là `DeepMimicEnv._reset_char(...)` dòng `151-157`

### Flow reset thật của DeepMimic

1. `_reset_ref_motion(env_ids)` dòng `175-197`
2. `_ref_state_init(env_ids)` dòng `202-215`
3. nếu visualize ref char, `_reset_ref_char(env_ids)` dòng `159-173`

### Ý nghĩa logic

Khác `CharEnv` thường, `DeepMimicEnv` không reset humanoid về một pose tĩnh đơn thuần.

Nó reset theo reference motion:

- sample motion id và time
- tính frame reference tại thời điểm đó
- set root/dof state của humanoid chính theo frame reference

Nghĩa là:

- train có thể bắt đầu từ nhiều điểm khác nhau trong motion clip
- nhờ vậy việc imitation ổn định hơn và không chỉ học từ frame đầu

---

## 22. Sau khi build xong, `step(action)` sẽ đi flow nào

File: [`mimickit/envs/sim_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/sim_env.py)

- `step()` ở dòng `77-89`

Flow:

1. `_pre_physics_step(action)` dòng `79`
2. `_physics_step()` dòng `81`
3. `_post_physics_step()` dòng `84`
4. nếu visualize thì render dòng `86-88`

## 22.1. `_pre_physics_step(action)`

- `SimEnv._pre_physics_step()` dòng `108-110`
- gọi `self._apply_action(actions)`

Với case này:

- `CharEnv._apply_action(...)` dòng `327-331`
- clamp action vào action bounds
- gọi `self._engine.set_cmd(char_id, clip_action)`

Trong engine:

- `IsaacLabEngine.set_cmd(...)` dòng `181-198`
- vì `control_mode = pos`
- dùng `obj.set_joint_position_target(sim_cmd)` dòng `188-189`

### Ý nghĩa logic

Action của PPO cuối cùng trở thành:

- target joint positions cho humanoid trong Isaac Lab

## 22.2. `_physics_step()`

- `SimEnv._physics_step()` dòng `112-114`
- gọi `self._step_sim()`
- `self._step_sim()` dòng `132-134`
- gọi `self._engine.step()`

Trong engine:

- `IsaacLabEngine.step()` dòng `136-148`

Flow:

1. update các object cần reset
2. nếu record video thì capture frame
3. loop `for i in range(self._sim_steps)` dòng `142`
4. mỗi vòng gọi:
   - `_pre_sim_step()`
   - `_sim_step()`
   - `_post_sim_step()`
5. cuối cùng clear forces

Với case này:

- `self._sim_steps = 4`
- nên 1 action control sẽ đi qua 4 physics substeps

## 22.3. `_post_physics_step()`

- `SimEnv._post_physics_step()` dòng `171-178`

Flow:

1. `_update_time()`
2. `_update_misc()`
3. `_update_observations()`
4. `_update_info()`
5. `_update_reward()`
6. `_update_done()`

Đây là chỗ env biến simulator state thành dữ liệu RL.

---

## 23. `DeepMimicEnv` tính observation thế nào

Trong [`deepmimic_env.py`](/home/wanhin/hienhq/MimicKit/mimickit/envs/deepmimic_env.py):

- `_compute_obs(...)` dòng `330-406`

### Flow thật

1. lấy motion id và motion time hiện tại
2. đọc state hiện tại từ engine:
   - root pos/rot/vel
   - dof pos/vel
   - body pos
3. đổi `dof_pos -> joint_rot`
4. nếu bật `phase_obs` thì tính phase
5. nếu có `key_bodies` thì lấy key body positions
6. nếu bật `tar_obs` thì lấy future target frames qua `_fetch_tar_obs_data(...)` dòng `563-581`
7. gọi `compute_deepmimic_obs(...)` dòng `389-405`

### Ý nghĩa logic

Observation của DeepMimic không chỉ là trạng thái hiện tại của humanoid.

Nó còn có thể chứa:

- phase trong motion
- target pose/position ở vài bước tương lai

Với config hiện tại:

- `enable_phase_obs = False`
- `enable_tar_obs = True`
- `tar_obs_steps = [1, 2, 3]`

nên obs sẽ có phần target future motion, nhưng không có phase encoding.

---

## 24. `DeepMimicEnv` tính reward thế nào

- `_update_reward()` dòng `408-461`
- `compute_reward(...)` dòng `787-846`

### Flow

1. lấy state hiện tại từ engine
2. đổi `dof_pos -> joint_rot`
3. lấy `key_pos` hiện tại
4. lấy reference state đã cập nhật sẵn:
   - `_ref_root_pos`
   - `_ref_root_rot`
   - `_ref_joint_rot`
   - `_ref_dof_vel`
   - `ref_key_pos`
5. gọi `compute_reward(...)`

### Bên trong `compute_reward(...)`

Reward gồm 5 thành phần:

- `pose_r`
- `vel_r`
- `root_pose_r`
- `root_vel_r`
- `key_pos_r`

Rồi cộng có trọng số:

```txt
r =
  pose_w * pose_r
  + vel_w * vel_r
  + root_pose_w * root_pose_r
  + root_vel_w * root_vel_r
  + key_pos_w * key_pos_r
```

### Ý nghĩa thật

Reward đang trả lời câu hỏi:

- humanoid hiện tại có giống motion reference không
- giống ở pose joint không
- giống ở velocity không
- giống ở root motion không
- giống ở các key body quan trọng không

---

## 25. `DeepMimicEnv` tính done thế nào

- `_update_done()` dòng `463-493`
- `compute_done(...)` dòng `724-784`

### Flow

1. lấy `motion_times`
2. lấy `motion_len`
3. lấy `motion_loop_mode`
4. lấy `root_rot`, `body_pos`, `ground_contact_forces` từ engine
5. gọi `compute_done(...)`

### `compute_done(...)` quyết định các kiểu done gì

- `TIME`: hết `episode_length`
- `SUCC`: motion clip kết thúc nếu clip không loop
- `FAIL`: ngã hoặc lệch pose quá xa nếu early termination bật

### Case hiện tại với config

- `enable_early_termination = True`
- `pose_termination = True`
- `pose_termination_dist = 1.0`
- `contact_bodies = ["right_foot", "left_foot"]`

Nghĩa là:

- nếu body nào khác chân chạm ground mạnh -> có thể fail
- nếu pose lệch reference quá xa -> có thể fail
- nếu motion clip kết thúc hợp lệ -> success

---

## 26. Tóm tắt đúng bản chất của `env_builder.build_env` trong case này

Nếu rút gọn lại đúng bản chất:

1. `run.py` đưa `env_config` và `engine_config` xuống `env_builder`
2. `env_builder` nhìn `env_name = "deepmimic"` rồi chọn `DeepMimicEnv`
3. `DeepMimicEnv` đọc các config imitation đặc thù
4. `SimEnv` build `IsaacLabEngine`
5. `CharEnv` load humanoid kinematic model và đăng ký humanoid object cho từng env
6. `IsaacLabEngine.initialize_sim()` mới thực sự dựng simulator/object/sensor
7. sau đó env mới build action space, tensor, buffer
8. từ đây PPO mới có một env hoàn chỉnh để `reset()`, `step()`, rollout và train

---

## 27. Cách nhớ 1 câu cho toàn flow

`env_builder.build_env` không chỉ “chọn class env”, mà với case `Isaac Lab + DeepMimic + humanoid` nó là điểm khởi đầu của cả chuỗi dựng `DeepMimicEnv`, load humanoid kinematic model, chọn `IsaacLabEngine`, đăng ký character cho từng env, khởi tạo simulator thật, rồi nối toàn bộ state simulator thành observation/reward/done để PPO dùng.`
