# G1 Assets Guide

Tài liệu này giải thích các file trong thư mục [data/assets/g1](/home/quanghien/hienhq/MimicKit/data/assets/g1).

## 1. Thư mục này chứa gì

`data/assets/g1/` là bộ asset của robot humanoid Unitree G1 trong repo này.

Nó gồm 5 nhóm nội dung chính:

- `g1.xml`: model MuJoCo chính để mô tả thân người, khớp, giới hạn khớp, quán tính, collision, actuator range
- `g1_mesh.xml`: biến thể MuJoCo thiên về mesh/visual để hiển thị hình dáng robot
- `g1.usd`: asset USD dùng cho pipeline/backend hỗ trợ USD như Isaac Lab
- `meshes/*.STL`: mesh 3D của từng link
- `LICENSE.txt`: license của asset G1 từ Unitree

## 2. Ý nghĩa từng file

### 2.1. `g1.xml`

File: [g1.xml](/home/quanghien/hienhq/MimicKit/data/assets/g1/g1.xml:1)

Đây là file quan trọng nhất nếu bạn đang đọc flow env hoặc physics.

Nó định nghĩa:

- tên model: `g1_29dof`
- các mesh sẽ dùng từ thư mục `meshes/`
- cây body-link của robot, bắt đầu từ `pelvis`
- `freejoint name="root"` cho root tự do trong không gian
- từng `joint` của chân, eo, tay
- giới hạn khớp `range`
- lực actuator `actuatorfrcrange`
- độ cứng `stiffness`, cản `damping`, `armature`
- quán tính và khối lượng từng link qua `inertial`
- collision shapes đơn giản như `sphere`, `capsule`

Hiểu ngắn gọn:

- nếu hỏi “robot G1 có những khớp nào, khớp quay theo trục nào, giới hạn bao nhiêu, va chạm ra sao”, thì câu trả lời chủ yếu nằm ở `g1.xml`

### 2.2. `g1_mesh.xml`

File: [g1_mesh.xml](/home/quanghien/hienhq/MimicKit/data/assets/g1/g1_mesh.xml:1)

Đây cũng là file MuJoCo, nhưng thiên về biểu diễn mesh trực quan của robot.

Nó vẫn có:

- cấu trúc body/joint tương tự G1
- inertial và joint ranges
- mesh của từng link

Điểm dễ thấy là file này thêm nhiều `geom type="mesh"` lặp lại để hiển thị đẹp hơn, và ở một số chỗ collision/marker đơn giản hơn hoặc khác cách mô tả so với `g1.xml`.

Hiểu ngắn gọn:

- `g1.xml`: thực dụng hơn cho physics/collision
- `g1_mesh.xml`: dễ dùng khi muốn nhìn hình robot rõ hơn theo mesh

### 2.3. `g1.usd`

File: [g1.usd](/home/quanghien/hienhq/MimicKit/data/assets/g1/g1.usd:1)

Đây là asset ở định dạng USD.

USD thường được dùng trong pipeline simulator/render hiện đại, đặc biệt khi backend cần asset dạng USD thay vì MJCF/XML.

Hiểu ngắn gọn:

- nếu backend đang load `.usd`, nó sẽ dùng file này
- nếu backend đang load `.xml`, nó sẽ dùng `g1.xml`

### 2.4. `meshes/`

Thư mục: [meshes](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes)

Đây là nơi chứa mesh STL của từng bộ phận robot.

Ví dụ:

- [pelvis.STL](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes/pelvis.STL)
- [left_hip_pitch_link.STL](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes/left_hip_pitch_link.STL)
- [right_knee_link.STL](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes/right_knee_link.STL)
- [torso_link.STL](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes/torso_link.STL)
- [head_link.STL](/home/quanghien/hienhq/MimicKit/data/assets/g1/meshes/head_link.STL)

Các file này không tự nói lên physics.

Chúng chủ yếu cung cấp:

- hình học bề mặt để render
- hình dáng trực quan của từng link
- dữ liệu mesh để `g1.xml` hoặc `g1_mesh.xml` tham chiếu tới

Tên file mesh thường map khá sát với tên body/link trong XML.

Ví dụ:

- body `left_hip_pitch_link` trong XML thường dùng mesh `left_hip_pitch_link.STL`
- body `right_wrist_yaw_link` thường dùng mesh `right_wrist_yaw_link.STL`

### 2.5. `LICENSE.txt`

File: [LICENSE.txt](/home/quanghien/hienhq/MimicKit/data/assets/g1/LICENSE.txt:1)

Đây là license BSD 3-Clause của asset từ Unitree.

Ý nghĩa:

- cho biết điều kiện sử dụng, phân phối lại, chỉnh sửa asset
- nếu bạn định mang asset này sang project khác hoặc public lại, nên đọc file này trước

## 3. Cách các file liên kết với nhau

Flow dễ hiểu nhất là:

1. Env config trỏ tới `char_file`, ví dụ `data/assets/g1/g1.xml`
2. `CharEnv` hoặc `DeepMimicEnv` đưa `char_file` đó cho engine
3. Engine/backend load file XML
4. Trong XML, phần `<asset>` tham chiếu tới các file `meshes/*.STL`
5. Phần `<worldbody>` và các `joint` mô tả cây khớp của robot
6. Physics chạy dựa trên body, joint, inertial, collision, actuator range trong XML

Nói ngắn:

- `meshes/` cho robot “hình dáng”
- `g1.xml` cho robot “cấu trúc và physics”
- `g1.usd` cho robot “định dạng asset khác để backend phù hợp dùng”

## 4. G1 trong repo này có những phần cơ thể nào

Nhìn theo tên mesh và body trong XML, G1 ở repo này gồm các nhóm chính:

- thân dưới:
  `pelvis`, `left/right_hip_pitch`, `hip_roll`, `hip_yaw`, `knee`, `ankle_pitch`, `ankle_roll`
- eo và thân:
  `waist_yaw`, `waist_roll`, `waist_pitch`, `torso`, `waist_support`, `logo`
- đầu:
  `head_link`
- tay trái/phải:
  `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `wrist_roll`, `wrist_pitch`, `wrist_yaw`, `rubber_hand`

Từ tên model `g1_29dof`, có thể hiểu đây là cấu hình G1 khoảng 29 bậc tự do trong asset này.

## 5. Khi nào nên đọc file nào

- Muốn hiểu robot spawn vào env thế nào: đọc `g1.xml`
- Muốn hiểu link nào có mesh gì: đọc thư mục `meshes/`
- Muốn xem asset cho USD backend: đọc `g1.usd`
- Muốn xem bản visual-heavy của model MuJoCo: đọc `g1_mesh.xml`
- Muốn kiểm tra quyền sử dụng asset: đọc `LICENSE.txt`

## 6. Một vài điểm dễ nhầm

### `g1.xml` không phải chỉ là file hình học

Sai.

Nó còn chứa:

- cấu trúc cây khớp
- khối lượng
- quán tính
- giới hạn khớp
- thông số actuator
- collision shapes

### `meshes/*.STL` không quyết định toàn bộ physics

Đúng hơn là:

- mesh chủ yếu để hiển thị
- physics/contact thường còn phụ thuộc các `geom` collision đơn giản trong XML như capsule, sphere

### `g1.usd` không thay thế hoàn toàn `g1.xml`

Hai file phục vụ hai kiểu backend/pipeline khác nhau.

- backend đọc MJCF/XML sẽ dùng `g1.xml`
- backend đọc USD sẽ dùng `g1.usd`

## 7. Ví dụ liên hệ với config env

Ví dụ các env G1 trong repo đang trỏ tới:

- [data/envs/deepmimic_g1_env.yaml](/home/quanghien/hienhq/MimicKit/data/envs/deepmimic_g1_env.yaml:3)
- [data/envs/amp_g1_env.yaml](/home/quanghien/hienhq/MimicKit/data/envs/amp_g1_env.yaml:3)
- [data/envs/add_g1_env.yaml](/home/quanghien/hienhq/MimicKit/data/envs/add_g1_env.yaml:3)

Các file này đều dùng:

```yaml
char_file: "data/assets/g1/g1.xml"
```

Nghĩa là trong flow train hiện tại của các env G1, file trung tâm vẫn là `g1.xml`.
