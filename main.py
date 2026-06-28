import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt  # Thêm thư viện vẽ biểu đồ

# ====================== CẤU HÌNH ĐIỀU KHIỂN DUY NHẤT 1 DÒNG ======================
# 🟢 LÚC MUỐN CHẠY 10 VIDEO: Bạn chỉ cần để trống như thế này: "" hoặc điền "all"
# 🔴 LÚC MUỐN CHẠY 1 VIDEO:  Bạn điền tên video vào đây (Ví dụ: "nhom7" hoặc "nhom1.2")
TU_KHOA_VIDEO = "nhom4.2"

FOLDER_PATH = "danh_sach_video"
# ================================================================================

print(f"--- HỆ THỐNG KIỂM ĐỊNH MẶT ĐƯỜNG NÂNG CAO - PCI ASTM D6433 ---")

if not os.path.exists(FOLDER_PATH):
    print(f"❌ LỖI: Không tìm thấy thư mục '{FOLDER_PATH}'!")
    exit()

all_mp4_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.mp4')]
all_mp4_files.sort()
video_list_to_process = []
chado_chay_le = False

if TU_KHOA_VIDEO.strip() == "" or TU_KHOA_VIDEO.lower() == "all":
    video_list_to_process = [os.path.join(FOLDER_PATH, f) for f in all_mp4_files[:10]]
    print(
        f"📂 [CHẾ ĐỘ TỰ ĐỘNG]: Nhận diện lệnh chạy HÀNG LOẠT. Đang nạp {len(video_list_to_process)} video để xử lý tuần tự...")
else:
    chado_chay_le = True
    normalized_keyword = TU_KHOA_VIDEO.replace(" ", "").lower()
    for file in all_mp4_files:
        if normalized_keyword in file.replace(" ", "").lower():
            video_list_to_process.append(os.path.join(FOLDER_PATH, file))
            break
    if len(video_list_to_process) == 0:
        print(f"❌ LỖI: Không tìm thấy video nào khớp với từ khóa '{TU_KHOA_VIDEO}'!")
        exit()
    print(
        f"🎯 [CHẾ ĐỘ TỰ ĐỘNG]: Nhận diện lệnh chạy RIÊNG LẺ. Video mục tiêu: {os.path.basename(video_list_to_process[0])}")

device_hardware = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo11n-seg.pt")

# ==================== THÔNG SỐ CẤU HÌNH THỰC TẾ ====================
REAL_VIEW_WIDTH_METER = 3.75
REAL_VIEW_LENGTH_METER = 6.0
FRAME_ROAD_AREA_M2 = REAL_VIEW_WIDTH_METER * REAL_VIEW_LENGTH_METER

CLASS_NUT_LUOI = 0
CLASS_NUT_DON = 1
CLASS_KHE_NOI = 2
# ===================================================================

all_segments_data = []
total_stt = 1

for v_idx, video_full_path in enumerate(video_list_to_process):
    video_name = os.path.basename(video_full_path)
    cap = cv2.VideoCapture(video_full_path)

    if not cap.isOpened(): continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pixel_to_meter = REAL_VIEW_WIDTH_METER / w
    frames_per_segment = total_frames // 20

    roi_points = np.array([
        [int(w * 0.05), int(h * 0.95)], [int(w * 0.25), int(h * 0.45)],
        [int(w * 0.75), int(h * 0.45)], [int(w * 0.95), int(h * 0.95)]
    ], dtype=np.int32)

    print(f"\n🚀 [KÍCH HOẠT] -> Đang phân tích chỉ số PCI cho: {video_name}")

    for seg_idx in range(20):
        frame_grid_areas = []
        frame_single_areas = []

        sample_count = 20
        start_frame = seg_idx * frames_per_segment
        frame_indices = np.linspace(start_frame, start_frame + frames_per_segment - 1, sample_count, dtype=int)

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            success, im0 = cap.read()
            if not success: break

            current_grid_m2 = 0.0
            current_single_m2 = 0.0

            results = model(im0, imgsz=320, device=device_hardware, verbose=False)[0]

            if results.masks is not None:
                for mask, box in zip(results.masks.xy, results.boxes):
                    cls = int(box.cls[0])
                    if cls == CLASS_KHE_NOI: continue
                    if cls == CLASS_NUT_LUOI and len(mask) > 0:
                        pixel_area = cv2.contourArea(np.array(mask, dtype=np.int32))
                        current_grid_m2 += pixel_area * (pixel_to_meter ** 2)
                    elif cls == CLASS_NUT_DON:
                        xywh = box.xywh[0].tolist()
                        real_length_meter = max(xywh[2], xywh[3]) * pixel_to_meter
                        current_single_m2 += real_length_meter * 0.3

            frame_grid_areas.append(current_grid_m2)
            frame_single_areas.append(current_single_m2)

            pct_total_now = min(((current_grid_m2 + current_single_m2) / FRAME_ROAD_AREA_M2) * 100, 100.0)
            pci_now = round(max(0.0, 100.0 - pct_total_now), 1)

            annotated_frame = results.plot()
            cv2.rectangle(annotated_frame, (10, 10), (480, 125), (15, 27, 42), -1)
            cv2.putText(annotated_frame,
                        f"Video: {v_idx + 1}/{len(video_list_to_process)} | Doan 50m: {seg_idx + 1}/20", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_frame,
                        f"S_nut luoi: {round(current_grid_m2, 2)} m2 | S_nut don: {round(current_single_m2, 2)} m2",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Diem PCI hien tai: {pci_now} Diem", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if pci_now >= 70 else (0, 165, 255), 2)

            preview_w, preview_h = 854, 480
            resized_preview = cv2.resize(annotated_frame, (preview_w, preview_h))

            scale_x, scale_y = preview_w / w, preview_h / h
            roi_preview = np.array([[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in roi_points], dtype=np.int32)
            cv2.polylines(resized_preview, [roi_preview], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("He thong theo doi vet nut mat duong", resized_preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        seg_avg_grid_m2 = np.mean(frame_grid_areas) if frame_grid_areas else 0.0
        seg_avg_single_m2 = np.mean(frame_single_areas) if frame_single_areas else 0.0
        total_crack_area_m2 = seg_avg_grid_m2 + seg_avg_single_m2
        total_road_scanned_m2 = FRAME_ROAD_AREA_M2

        pct_grid = (seg_avg_grid_m2 / total_road_scanned_m2) * 100
        pct_single = (seg_avg_single_m2 / total_road_scanned_m2) * 100
        seg_pct_total = min(pct_grid + pct_single, 100.0)

        diem_tru_phat = (pct_grid * 3.0) + (pct_single * 1.5)
        pci_score = round(max(0.0, 100.0 - diem_tru_phat), 2)

        if 85.0 <= pci_score <= 100.0:
            pavement_condition = "Good (Tot)"
        elif 70.0 <= pci_score < 85.0:
            pavement_condition = "Satisfactory (Hai long)"
        elif 55.0 <= pci_score < 70.0:
            pavement_condition = "Fair (Trung binh)"
        elif 40.0 <= pci_score < 55.0:
            pavement_condition = "Poor (Kem)"
        elif 25.0 <= pci_score < 40.0:
            pavement_condition = "Very Poor (Rat kem)"
        elif 10.0 <= pci_score < 25.0:
            pavement_condition = "Serious (Nghiem trong)"
        else:
            pavement_condition = "Failed (Hu hong hoan toan)"

        start_m = seg_idx * 50
        end_m = (seg_idx + 1) * 50

        all_segments_data.append({
            "STT": total_stt,
            "Ten file video": video_name,
            "Phan doan": f"{start_m}m - {end_m}m",
            "Dien tich nut luoi (m2)": round(seg_avg_grid_m2, 3),
            "Dien tich nut don (m2)": round(seg_avg_single_m2, 3),
            "Tong dien tich nut (m2)": round(total_crack_area_m2, 3),
            "Ti le dien tich nut (%)": round(seg_pct_total, 4),
            "Chi so PCI (0-100)": pci_score,
            "Tinh trang mat duong (ASTM D6433)": pavement_condition
        })
        total_stt += 1

    print(f"   > Da phan tich xong chi so PCI cho file: {video_name}")
    cap.release()

cv2.destroyAllWindows()

# ==================== ĐOẠN XUẤT BÁO CÁO TỰ ĐỘNG THÔNG MINH ====================
if len(all_segments_data) > 0:
    if not chado_chay_le:
        output_csv = "ket_qua_200_phan_doan_tong_hop.csv"
        output_img = "bieu_do_tong_hop_200_phan_doan.png"
    else:
        video_base_name = os.path.basename(video_list_to_process[0]).replace('.mp4', '')
        output_csv = f"bao_cao_pci_{video_base_name}.csv"
        output_img = f"bieu_do_pci_{video_base_name}.png"

    df = pd.DataFrame(all_segments_data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("\n================ TÍNH TOÁN PCI HOÀN TẤT ================")
    print(f"🎉 Da xuat file bao cao bieu mau tai: '{output_csv}'")

    # ==================== 🔥 ĐOẠN CODE TỰ ĐỘNG VẼ BIỂU ĐỒ NÂNG CAO ====================
    print("[*] Đang khởi tạo sơ đồ trực quan hóa dữ liệu...")

    # Để tránh đồ thị bị quá dày khi chạy 10 video (200 dòng), ta giới hạn hiển thị 20 đoạn đại diện trên biểu đồ công bố
    df_plot = df.head(20)

    # Cấu hình khung nền vẽ (Gồm 2 biểu đồ chồng lên nhau)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f"BÁO CÁO TRỰC QUAN DIỄN BIẾN KHUYẾT TẬT VÀ PCI MAT ĐƯỜNG\nNguồn: {TU_KHOA_VIDEO if chado_chay_le else 'Danh sach 10 Video tổng hợp'}",
        fontsize=14, fontweight='bold', color='#2c3e50', y=0.97)

    # 📊 BIỂU ĐỒ 1: Biểu đồ cột chồng thể hiện Tỉ lệ diện tích nứt (%)
    # Tính tỉ lệ % nứt đơn và nứt lưới riêng biệt cho các mẫu vẽ
    pct_grid_series = (df_plot["Dien tich nut luoi (m2)"] / FRAME_ROAD_AREA_M2) * 100
    pct_single_series = (df_plot["Dien tich nut don (m2)"] / FRAME_ROAD_AREA_M2) * 100

    ax1.bar(df_plot["Phan doan"], pct_grid_series, label="Ti le nut luoi (%)", color='#e74c3c', alpha=0.85)
    ax1.bar(df_plot["Phan doan"], pct_single_series, bottom=pct_grid_series, label="Ti le nut don (%)", color='#f39c12',
            alpha=0.85)

    ax1.set_ylabel("Ti le nut / Dien tich (%)", fontsize=11, fontweight='bold')
    ax1.set_title("1. Sơ đồ phân bổ tỉ lệ diện tích khuyết tật bề mặt theo phân đoạn 50m", fontsize=12,
                  fontweight='bold', color='#34495e', loc='left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # 📈 BIỂU ĐỒ 2: Biểu đồ đường thể hiện Chỉ số PCI (ASTM D6433)
    ax2.plot(df_plot["Phan doan"], df_plot["Chi so PCI (0-100)"], marker='o', color='#27ae60', linewidth=2.5,
             label="Chi so PCI thuc te")

    # Đổ dải màu nền cảnh báo chất lượng theo tiêu chuẩn
    ax2.axhspan(85, 100, color='#2ecc71', alpha=0.15, label="Tot (85-100)")
    ax2.axhspan(55, 85, color='#f1c40f', alpha=0.15, label="Trung binh (55-85)")
    ax2.axhspan(0, 55, color='#e74c3c', alpha=0.15, label="Kem / Nguy hiem (<55)")

    ax2.set_xlabel("Vi tri phân doan hình học (mét)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Diem so PCI (0-100)", fontsize=11, fontweight='bold')
    ax2.set_title("2. Sơ đồ biến thiên chỉ số tình trạng mặt đường PCI (Pavement Condition Index)", fontsize=12,
                  fontweight='bold', color='#34495e', loc='left')
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', ncol=4)

    # Hiển thị điểm số PCI dạng chữ số ngay trên các đầu nút tọa độ
    for idx, val in enumerate(df_plot["Chi so PCI (0-100)"]):
        ax2.text(idx, val + 2, f"{int(val)}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#117a65')

    # Định dạng trục tọa độ, xoay nhãn 45 độ tránh đè chữ
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Tự động xuất ảnh đồ thị độ nét cao để chèn phụ lục báo cáo
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"📊 [ĐỒ THỊ]: Đã xuất ảnh biểu đồ phân tích thành công tại: '{output_img}'")

    # Hiển thị trực quan cửa sổ biểu đồ lên màn hình máy tính
    plt.show()