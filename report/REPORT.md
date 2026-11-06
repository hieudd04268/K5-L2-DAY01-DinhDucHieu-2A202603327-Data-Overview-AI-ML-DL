# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên, không đổi)

**Thay đổi so với notebook nguồn:** Không đổi checkpoint, threshold hay mã tạo prediction. Chỉ thêm 1 ô đọc JSON để trích số liệu cho báo cáo.

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy` hay `polygon_xy`, chỉ có một lớp được xếp hạng cho toàn ảnh 640×428. Model không chỉ ra chiếc "cab" nằm ở đâu, và top-5 (cab, minibus, police_van, recreational_vehicle, streetcar) là 5 cách mô tả khác nhau cho cùng một ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Class list **đi kèm checkpoint/taxonomy**: `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K (1000 lớp), nên model chỉ chọn trong danh sách đó. ImageNet không có lớp "traffic" hay "street scene", nên model phải chọn lớp gần nhất là `cab`.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Cùng một tên lớp có thể có ID khác hoặc nghĩa khác ở taxonomy khác. Ví dụ `cab` (id 468) là lớp của ImageNet-1K, còn COCO-80 không có lớp `cab` mà chỉ có `car`/`bus`. Ghi đủ ba trường thì nhãn không bị hiểu nhầm, máy đọc được và đối chiếu được.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline phải nói rõ: (a) nhãn mô tả **chủ thể chính** hay **cảnh tổng thể**; (b) tiêu chí chọn chủ thể chính (chiếm diện tích lớn nhất / ở tiền cảnh / ở trung tâm); (c) dự án là single-label hay multi-label; (d) khi không có lớp phù hợp hoặc hai lớp ngang nhau thì escalation. Ảnh `traffic` có xe buýt, ô tô con, người đi bộ... nếu không có quy tắc thì mỗi annotator sẽ chọn một lớp khác nhau.

6. Vì sao không được lấy `score` của model làm ground truth?

   Score chỉ là mức "tin" của model, không phải sự thật. Ở đây `cab` chỉ có 0.51, tức là model khá phân vân, và trên ảnh, vật nổi bật ở tiền cảnh là **xe buýt xanh**, không thấy rõ taxi nào. Model có thể tin cao mà vẫn sai. Ground truth phải do người gán theo guideline và được reviewer kiểm tra.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person"; `score`=0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? 
Lớp "person", score 0.91; góc trên-trái khoảng pixel (385.33, 69.24), góc dưới-phải khoảng pixel (498.92, 348.92); rộng khoảng 114px và cao khoảng 280px
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ví dụ so hai ngưỡng trên ảnh `kitchen`: threshold 0.20 → nhiều prediction hơn (vd. 17); threshold 0.60 → ít hơn (vd. 6). 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? 
Khi hạ ngưỡng từ 0.60 xuống 0.20, số hộp cần kiểm tra tăng từ 6 lên 11 (thêm 5 hộp). Các hộp thêm vào có score thấp (0.38–0.50) nên dễ là nhiễu/false positive. Vì vậy reviewer phải xem nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng (khoảng 83%, tức 11/6).
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). 
Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? 
- oven box (0.11, 187.54) → (195.96, 292.81): x1 ≈ 0.11 sát biên trái, tức object chạm mép khung hình → có nguy cơ bị cắt.
- person box (0.12, 263.18) → (61.55, 310.78) và bowl box (0.32, 305.02) → (87.69, 371.32): cũng chạm biên trái.
- oven box (489.09, 200.73) → (617.81, 344.88): gần biên phải (ảnh rộng 640) nhưng x2 = 617.81 < 640 nên chưa vượt khung.
- Không box nào vượt khỏi ảnh (x ≥ 0, y ≥ 0, x2 ≤ 640, y2 ≤ 427) — các box sát biên là do object bị cắt bởi mép ảnh, không phải box vẽ lỗi.
--> Cách xử lý: Tôi theo guideline đã có — nếu object còn ≥ 50% trong khung thì vẫn gán box ôm sát phần nhìn thấy (không nới box ra ngoài ảnh); nếu < 50% hoặc bị che khuất nặng, không chắc là object gì thì escalation hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "traffic-001"; `class_name`= "bus"; `score`=0.925745; số điểm polygon=`polygon_point_count`=120; 5 điểm đầu của `polygon_xy`= (148,189), (147,190), (145,190), (143,192), (142,192)
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ cho vùng chữ nhật bao quanh (kèm cả nền ở 4 góc), còn polygon cho hình dạng thật/đường viền của vật thể: chỗ lồi lõm, phần bị che, ranh giới giữa hai vật chạm nhau. Box không phân biệt được hai xe đỗ sát nhau, polygon thì tách được nhờ mask bám mép.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id định danh một vật thể cụ thể trong một ảnh, để nối bbox, polygon và metadata của cùng một vật (vd. traffic-001 là đúng chiếc bus đó, không lẫn với traffic-005).
- Không phải class ID: class ID (class_id = 5) chỉ nói loại (xe bus), dùng chung cho mọi chiếc bus; instance_id nói cá thể nào.
- Không phải tracking ID: tracking ID giữ nguyên một vật qua nhiều frame/theo thời gian; instance_id chỉ có ý nghĩa trong một ảnh, sang ảnh khác sẽ cấp lại.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc vẽ biên mask: Vẽ đường biên bám sát mép phần nhìn thấy của vật thể, không lấn sang nền/vật khác; khi hai vật chạm nhau phải vẽ ranh giới theo đúng điểm tiếp xúc (không gộp chung một mask); phần bị che thì dừng ở mép vật che, không tự bịa đường viền phía sau.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Trên ảnh kitchen, chỗ chạm/đè rõ nhất là vùng mặt bàn: mask dining table (instance_id = kitchen-005, score 0.5784, 598 điểm polygon, bbox (0.97, 234.74) → (345.86, 423.87)) phủ lên và chạm nhiều vật đặt trên bàn — đáng chú ý là hai chiếc bowl kitchen-002 (31.44, 343.77) → (100.89, 385.55) và kitchen-006 (0.11, 302.85) → (85.7, 370.15) chồng lấn nhau, cạnh đó còn kitchen-003 (bowl) và các spoon.
- Xử lý: trước hết theo guideline. Nếu guideline chưa nói rõ cách tách biên giữa bàn với đồ vật đặt trên bàn (và giữa hai bowl chạm nhau) thì escalation hỏi Lab Coach, không tự bịa đường tách. Nếu guideline đã quy định "tách theo khe nhìn thấy giữa hai object" thì làm theo guideline: vẽ biên dừng đúng ở khe hở nhìn thấy, không gộp chung mask, không để mask bàn "nuốt" phần đồ vật.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn cho cả ảnh, theo taxonomy ImageNet-1K (chỉ class, không box/polygon) | Top-1 của model có thể lệch trực giác cảnh giao thông (vd. ảnh đường phố bị gán lớp gần nghĩa như bus/truck/streetcar, hoặc lớp nền) → prediction ≠ GT; ảnh có nhiều vật thì khó chọn 1 lớp đại diện | Đọc guideline, chọn đúng 1 lớp cho toàn ảnh theo vật thể chính/ngữ cảnh; không chép score thành nhãn | Đúng taxonomy, đúng phạm vi cấp ảnh, nhãn là class chứ không phải score, nhất quán giữa các annotator |
| Phát hiện vật thể | Mỗi object: 1 class + box xyxy (pixel) | (1) Hạ ngưỡng 0.60 → 0.20 trên kitchen: 6 → 11 hộp, thêm 5 hộp score thấp 0.38–0.50 (bowl 0.4999, bowl 0.4649, cup 0.4511, cup 0.3815, bowl 0.3812) → dễ là hộp nhiễu. (2) Vật cắt mép: oven (0.11,187.54)→(195.96,292.81) có x1≈0.11 sát biên trái; person (0.12,263.18)→(61.55,310.78) cũng sát trái. (3) Hai bowl gần nhau dễ sinh box trùng/chồng | Vẽ box ôm sát phần nhìn thấy, đủ object theo guideline; không gộp 2 object tách rời; vật cắt mép còn ≥50% trong khung thì vẫn gán | Box có ôm sát không; thiếu/thừa object; xử lý ca che khuất/cắt mép; có box trùng lặp không |
| Instance segmentation | Mỗi instance: 1 class + polygon (pixel) + instance_id riêng | (1) Biên hai vật chạm nhau: kitchen-005 dining table (598 điểm, bbox (0.97,234.74)→(345.86,423.87)) đè/chạm các bowl kitchen-002 (67 điểm) và kitchen-006 (84 điểm) chồng lấn → khó tách biên bàn vs đồ vật. (2) Vùng mờ/nhỏ: kitchen-010 bowl chỉ 12 điểm, score 0.3724 → rất khó xác định biên. (3) Mask dễ tràn sang object khác nếu vẽ ẩu | Vẽ mask bám biên từng instance, tách theo khe nhìn thấy, không gộp mask, dừng ở mép vật che; mỗi instance một instance_id | Mask có tràn object khác không; có đủ instance cùng lớp (vd. 4 bowl) không; biên chạm/che xử lý đúng chưa; instance_id có trùng/thiếu không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng thao tác ngay, không upload/chia sẻ, và báo Lab Coach / GV phụ trách lab để được hướng dẫn; chỉ tiếp tục khi có xác nhận dữ liệu hợp lệ.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
