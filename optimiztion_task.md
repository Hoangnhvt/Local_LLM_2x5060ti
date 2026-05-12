# PRD: Hệ Thống Home AI — Tối Ưu Hóa Toàn Diện

**Version:** 1.0
**Author:** kurts_po
**Date:** 2026-05-06
**Status:** Draft

---

## 1. Problem Statement

Hệ thống Home AI hiện tại có cấu hình phần cứng mạnh mẽ (i7-10700K, 32GB RAM, 2x RTX 5060 Ti) nhưng chưa được khai thác tối đa. Các vấn đề chính:

- **vLLM chạy hai instance song song** nhưng phân bổ tài nguyên chưa tối ưu (mỗi GPU dùng ~13.7GB/16GB — gần đầy, không còn dư địa chỉ VRAM cho inference nặng)
- **Hệ thống Hermes Agent** bị lỗi kết nối Telegram và lỗi API (HTTP 400/429/500) làm gián đoạn dịch vụ
- **FreeRDP/gnome-remote-desktop** gặp lỗi NLA authentication và connection timeout
- **Bảo mật hệ thống** đang yếu: chưa có fail2ban, UFW chưa kích hoạt, SSH/RDP mở hoàn toàn

**Ảnh hưởng:** Hiệu suất suy giảm, trải nghiệm người dùng không ổn định, lỗ hổng bảo mật tiềm tàng.

---

## 2. Goals & Success Metrics

**Mục tiêu chính:** Tối ưu hóa toàn diện hệ thống Home AI — phần cứng, phần mềm, và agent — để đạt hiệu suất cao nhất với tài nguyên hiện có.

**Chỉ số thành công:**
| Chỉ tiêu | Mục tiêu |
|-----------|----------|
| VRAM sử dụng mỗi GPU | ≤12GB (giữ ≥4GB reserve cho peak load) |
| Thời gian phản hồi agent | < 300ms cho lệnh đơn giản |
| Tỷ lệ lỗi kết nối Telegram | < 1% trong 24 giờ |
| Số lỗi API (429 rate limit) | Bằng 0 |
| Uptime hệ thống | ≥ 99.5% |
| Phản hồi RDP | < 50ms latency nội bộ |

**Không phải mục tiêu (v1):**
- Nâng cấp phần cứng mới
- Di chuyển sang nền tảng cloud
- Tích hợp thêm model AI ngoài vLLM

---

## 3. Phần Cứng Hiện Tại

### 3.1 CPU
- **Model:** Intel Core i7-10700K @ 3.80GHz
- **Cấu hình:** 8 nhân / 16 luồng
- **Cache:** 16MB L3
- **Hiệu suất:** Hỗ trợ AVX2, AES-NI, và các lệnh vector cho inference nhanh

### 3.2 RAM
- **Tổng:** 32GB DDR4
- **Sử dụng:** ~12GB (37.5%)
- **Còn lại:** ~18GB available

### 3.3 GPU
- **Model:** 2x NVIDIA GeForce RTX 5060 Ti (16GB GDDR6 mỗi card)
- **VRAM sử dụng:** ~13.7GB/GPU cho vLLM workers
- **Nhiệt độ:** GPU 0: 46°C, GPU 1: 42°C
- **Công suất:** ~25W/GPU (chế độ P1, tiêu thụ thấp)
- **CUDA:** Version 13.0, Driver 580.126.09

### 3.4 Mạng & Bảo Mật
- SSH (cổng 22): Mở hoàn toàn, đang bị brute force
- RDP (cổng 3389): Mở hoàn toàn qua FreeRDP
- Firewall: UFW cài đặt nhưng chưa kích hoạt
- Fail2ban: Chưa cài đặt

---

## 4. Tối Ưu Hệ Thống

### 4.1 Tối Ưu GPU & vLLM

**Trạng thái thực tế (2026-05-06):**
| GPU | VRAM Used | VRAM Total | Free | Temp | Power |
|-----|-----------|------------|------|------|-------|
| GPU 0 | **14,753 MB** (90.4%) | 16,311 MB | ~1,558 MB | 46°C | 24.8W |
| GPU 1 | **13,945 MB** (85.5%) | 16,311 MB | ~2,366 MB | 41°C | 8.2W |

> Cả hai GPU đang chạy `coder-lg-nvfp4` (Qwen3.6-27B NVFP4, TP=2) với `gpu_memory_utilization=0.92`.
> Reserve thực tế: GPU 0 chỉ còn **~1.5 GB** — thấp hơn nhiều so với mục tiêu ≥4 GB.

**Vấn đề:** `gpu-memory-utilization=0.92` để lại <2 GB headroom trên GPU 0. Để đạt ≥4 GB reserve trên cả hai GPU cần hạ xuống ~**0.73** (`12 GB / 16.311 GB`). Tuy nhiên với NVFP4 weights ~9.85 GB/GPU, mức 0.73 vẫn đủ tải model và để ~4.1 GB cho KV cache.

**Yêu cầu:**
| ID | Yêu cầu | Ưu tiên | Trạng thái |
|-----|---------|----------|------------|
| SYS-001 | Hạ `GPU_MEM_UTIL` từ 0.92 → **0.73** trong `.env` để giữ ≥4 GB reserve mỗi GPU | P0 | ⏳ Chưa làm |
| SYS-002 | `--tensor-parallel-size=2` đã áp dụng cho `coder-lg-nvfp4` — giữ nguyên | P0 | ✅ Done |
| SYS-003 | `--enable-prefix-caching` đã bật trong `coder-lg-nvfp4.yml` — giữ nguyên | P1 | ✅ Done |
| SYS-004 | Cấu hình `--swap-space 8` trong `coder-lg-nvfp4.yml` phòng khi VRAM đầy đột biến | P1 | ⏳ Chưa làm |
| SYS-005 | Monitor VRAM qua `nvidia-smi dmon` và cảnh báo khi vượt 12 GB/GPU | P2 | ⏳ Chưa làm |

**Phân tích VRAM per GPU (16,311 MB):**
| `gpu_memory_utilization` | Budget | KV Headroom | Kết luận |
|--------------------------|--------|-------------|----------|
| 0.73 | 11,907 MB | **557 MB** | ❌ Fail — model không serve được |
| 0.80 | 13,049 MB | 1,699 MB | ❌ Quá thấp — OOM khi request dài |
| 0.85 | 13,864 MB | 2,514 MB | ⚠️ Tối thiểu khả dụng |
| 0.88 | 14,354 MB | 3,004 MB | ✅ An toàn, ~32K context |
| **0.92** | **15,006 MB** | **3,656 MB** | ✅ Hiện tại — cần thiết |

> **Kết luận SYS-001:** Không thể giảm `GPU_MEM_UTIL` thấp hơn 0.85 với model này. Weights NVFP4 (~9.85 GB/GPU) + runtime (~1.5 GB) đã chiếm ~11.35 GB/GPU — hardware limit, không phải lỗi cấu hình.

**Hướng xử lý thực tế:**
- **Giữ `GPU_MEM_UTIL=0.92`** — đây là mức đúng cho model này
- **Không co-load model khác** khi `coder-lg-nvfp4` đang chạy
- **Monitor thay vì giảm util**: dùng `nvidia-smi dmon` alert khi free VRAM < 1 GB
- Nếu cần VRAM reserve lớn hơn → đổi sang model nhỏ hơn (`reason`, `coder-md`)

### 4.2 Tối Ưu CPU & RAM

| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| SYS-006 | Giới hạn số process Hermes để không vượt quá 60% CPU (10/16 threads) | P0 |
| SYS-007 | Cấu hình `nice`/`cpuset` cho vLLM workers để ưu tiên inference | P1 |
| SYS-008 | Kích hoạt swap file 16GB cho peak RAM usage | P1 |
| SYS-009 | Cấu hình `vm.swappiness=10` để ưu tiên RAM cho ứng dụng | P2 |

### 4.3 Bảo Mật Hệ Thống

| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| SEC-001 | Cài đặt và kích hoạt `fail2ban` cho SSH (khóa sau 3 lần thất bại, khóa 10 giờ) | P0 |
| SEC-002 | Kích hoạt `ufw` và giới hạn SSH/RDP chỉ cho mạng nội bộ (10.8.x.x, 192.168.x.x) | P0 |
| SEC-003 | Tắt đăng nhập root qua SSH (`PermitRootLogin no`) | P0 |
| SEC-004 | Cấu hình `gnome-remote-desktop` với mật khẩu mạnh và tắt NLA fallback | P1 |
| SEC-005 | Tạo firewall rule cho cổng vLLM API (thường là 8000) chỉ cho LAN | P1 |

---

## 5. Tối Ưu Phần Mềm

### 5.1 Hermes Agent & Gateway

**Vấn đề hiện tại:**
- Lỗi kết nối Telegram: `httpx.ConnectError` và `ReadError`
- Lỗi API: HTTP 400 (`auto` tool choice), HTTP 429 (rate limit), HTTP 500 (internal error)
- Bot `@kurts_pobot` cần viết file PRD trước khi gửi

**Yêu cầu:**
| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| SW-001 | Cấu hình Telegram gateway: thêm fallback IP cố định `149.154.167.220` và tăng retry count | P0 |
| SW-002 | Sửa lỗi `auto` tool choice: thêm `--enable-auto-tool-choice` và `--tool-call-parser` cho model provider | P0 |
| SW-003 | Cấu hình rate limit handling: exponential backoff + queue messages khi gặp HTTP 429 | P0 |
| SW-004 | Đảm bảo `@kurts_pobot` luôn dùng `write_file` → `verify` → `MEDIA:` workflow | P0 |
| SW-005 | Tối ưu session management: tự động dọn session cũ sau 7 ngày | P1 |
| SW-006 | Giảm kích thước context window bằng cách compact logs tự động | P1 |

### 5.2 FreeRDP / Remote Desktop

**Vấn đề:** NLA authentication thất bại (`Could not find user in SAM database`)

**Yêu cầu:**
| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| SW-007 | Thêm user `kurt` vào FreeRDP SAM: `freerdp-sam --add kurt --password <pwd>` | P0 |
| SW-008 | Hoặc chuyển sang VNC qua Ubuntu Settings → Sharing | P0 |
| SW-009 | Cấu hình connection timeout lớn hơn (từ 30s → 120s) để giảm lỗi `Connection timed out` | P1 |
| SW-010 | Monitor remote desktop session và tự động reconnect | P2 |

### 5.3 Hệ Thống Operative

| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| SW-011 | Cấu hình `systemd` service cho tất cả thành phần Home AI | P0 |
| SW-012 | Tự động restart service khi crash (`Restart=always`) | P0 |
| SW-013 | Log rotation: giới hạn log files ≤100MB, giữ 7 ngày | P1 |
| SW-014 | Tự động cập nhật an toàn: `unattended-upgrades` cho security patches | P1 |

---

## 6. Tối Ưu Agent

### 6.1 Tối Ưu Model Inference

**Vấn đề:** Model `coder-lg-nvfp4` gặp lỗi cấu hình và rate limit.

**Yêu cầu:**
| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| AGT-001 | Cấu hình `config.yaml`: thêm `tool_choice: "auto"` đúng cách với flags cần thiết | P0 |
| AGT-002 | Giảm `max_tokens` cho các task đơn giản để tiết kiệm tài nguyên | P1 |
| AGT-003 | Cấu hình fallback model: khi primary model lỗi, tự động chuyển sang model nhẹ hơn | P0 |
| AGT-004 | Tối ưu prompt template: giảm độ dài system prompt để giảm token usage | P1 |
| AGT-005 | Cấu hình concurrency limit cho agent sessions (tối đa 3 song song) | P1 |

### 6.2 Tối Ưu Performance Agent

| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| AGT-006 | Sử dụng `delegate_task` cho task song song thay vì tuần tự | P0 |
| AGT-007 | Cấu hình context compaction tự động khi vượt quá 8000 chars | P0 |
| AGT-008 | Giảm số tool calls không cần thiết bằng cách cache kết quả API | P1 |
| AGT-009 | Tối ưu skill loading: chỉ load skills liên quan thay vì tất cả | P1 |
| AGT-010 | Cấu hình memory management: tự động dọn memories cũ sau 30 ngày | P2 |

### 6.3 Tối Ưu Cron Jobs & Tự Động Hóa

| ID | Yêu cầu | Ưu tiên |
|-----|---------|----------|
| AGT-011 | Tối ưu cron jobs: giảm tần suất cho jobs không cần thiết | P1 |
| AGT-012 | Cấu hình job delivery: tự động gửi kết quả qua Telegram | P1 |
| AGT-013 | Tạo cron job giám sát hệ thống: kiểm tra CPU/RAM/VRAM mỗi 30 phút | P1 |

---

## 7. Kế Hoạch Triển Khai

### Phase 1: Bảo Mật & Ổn Định (Tuần 1)
- [ ] Cài đặt fail2ban + kích hoạt UFW (SEC-001, SEC-002)
- [ ] Sửa lỗi Telegram gateway (SW-001, SW-002)
- [ ] Sửa lỗi NLA FreeRDP (SW-007)
- [ ] Cấu hình vLLM tensor parallel (SYS-002, SYS-001)

### Phase 2: Tối Ưu Hiệu Suất (Tuần 2)
- [ ] Tối ưu VRAM reserve (SYS-001)
- [ ] Cấu hình CPU/RAM optimization (SYS-006, SYS-008)
- [ ] Tối ưu model & agent (AGT-001, AGT-003)
- [ ] Tự động hóa monitoring (AGT-013)

### Phase 3: Hoàn Thiện & Giám Sát (Tuần 3)
- [ ] Log rotation & service management (SW-011, SW-013)
- [ ] Cron jobs & tự động hóa (AGT-011, AGT-012)
- [ ] Testing end-to-end và validation
- [ ] Tài liệu hóa và bàn giao

---

## 8. Các Trường Hợp Đặc Biệt

| Tình huống | Phản hồi Hệ Thống |
|-------------|-------------------|
| VRAM đầy | vLLM offload sang swap (SYS-004) |
| Lỗi kết nối Telegram | Tự động reconnect với exponential backoff (SW-003) |
| Agent crash | Service tự động restart qua systemd (SW-012) |
| Rate limit API | Queue request và retry sau cooldown |
| Tấn công SSH | fail2ban tự động khóa IP (SEC-001) |

---

## 9. Open Questions

1. Nên dùng tensor parallel size 2 hay giữ hai model riêng rẽ?
2. Model nào nên là fallback khi `coder-lg-nvfp4` gặp lỗi?
3. Nên giữ FreeRDP hay chuyển hoàn toàn sang VNC?
4. Tần suất monitoring cron job: 30 phút có đủ?
5. Có cần thêm SSD cho swap file hay dùng RAM disk?

---

*PRD được tạo bởi kurts_po — Product Owner Agent*
