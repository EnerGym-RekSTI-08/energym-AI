# EnerGym AI — Edge AI Server

Server AI berbasis pose estimation yang berjalan di laptop gym station. Menerima koneksi dari aplikasi mobile via HTTP dan WebSocket, memproses video dari webcam secara real-time, menghitung rep, dan mendeteksi bad form otomatis.

---

## Arsitektur Sistem

```
+-----------------+        WiFi / LAN        +----------------------+
|  Mobile App     | ---- HTTP / WebSocket --> |  EnerGym AI Server   |
|  (React Native) |                           |  (FastAPI + Python)  |
+-----------------+                           +----------+-----------+
                                                         |
                                              +----------v-----------+
                                              |  Webcam (USB/Built-in)|
                                              |  MediaPipe Pose       |
                                              |  Exercise Analyzer    |
                                              +----------+-----------+
                                                         |
                                              +----------v-----------+
                                              |  Supabase (Cloud DB) |
                                              |  workout_sessions    |
                                              +----------------------+
```

---

## Fitur

- **Pose Estimation Real-time** — MediaPipe BlazePose untuk tracking 33 keypoint tubuh
- **Rep Counter Otomatis** — Deteksi gerakan naik/turun berdasarkan sudut sendi
- **Bad Form Detection** — Deteksi 4 jenis kesalahan postur:
  - `body_sway` — Badan berayun saat curl
  - `elbow_drift` — Siku bergerak maju
  - `too_fast` — Gerakan terlalu cepat
  - `grip_rotation` — Rotasi grip tidak netral (Hammer Curl)
- **Multi-set Management** — Tracking rep per set dengan offset
- **Double-buffer Snapshot** — Stream MJPEG ke mobile tanpa flicker
- **Supabase Sync** — Push hasil sesi ke cloud, offline queue jika koneksi putus
- **QR Generator** — Generate QR station dengan auto-increment dan simpan ke DB

---

## Exercise yang Didukung

| Exercise | Config Key | Keterangan |
|---|---|---|
| Bicep Curl | `bicep_curl` | Unilateral, siku kanan |
| Alternating Dumbbell Curl | `alternating_curl` | Bilateral bergantian |
| Hammer Curl | `hammer_curl` | Grip netral, cek wrist deviation |

---

## Struktur Project

```
energym-AI/
├── configs/
│   └── default.yaml          # Konfigurasi kamera, MediaPipe, exercise thresholds
├── scripts/
│   └── generate_qr.py        # Generator QR code untuk gym station
├── src/energym_ai/
│   ├── server.py             # FastAPI server (entry point utama)
│   ├── run.py                # Standalone pipeline (tanpa server, tampil di layar)
│   ├── core/
│   │   └── pose_detector.py  # Wrapper MediaPipe pose estimation
│   ├── exercises/
│   │   ├── bicep_curl.py
│   │   ├── alternating_curl.py
│   │   └── hammer_curl.py
│   ├── output/
│   │   ├── supabase_sync.py  # Push hasil sesi ke Supabase
│   │   ├── cloud_sync.py     # Generic cloud sync via HTTP
│   │   └── esp32_alert.py    # Serial alert ke ESP32 buzzer
│   └── utils/
│       └── config.py         # YAML config loader
├── .env                      # Credentials (tidak di-commit)
├── pyproject.toml
└── README.md
```

---

## Instalasi

### Prasyarat

- Python 3.10+
- Webcam (USB atau built-in)
- Laptop terhubung ke WiFi yang sama dengan HP

### Setup

```bash
# Clone dan masuk ke folder
cd energym-AI

# Buat virtual environment
python -m venv .venv

# Aktivasi (Windows)
.venv\Scripts\activate

# Install semua dependency
pip install -e .
```

### Konfigurasi `.env`

Buat file `.env` di root project:

```env
SUPABASE_URL=https://hzxjzksqxrosgnpernih.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh6eGp6a3NxeHJvc2ducGVybmloIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NzYyNzAwNSwiZXhwIjoyMDkzMjAzMDA1fQ.W2KTYcoute-awF1bGsenEekbfkT8bvpTUjrOKawVhyk   
ENERGYM_STATION_ID=STATION_01
```

---

## Menjalankan Server

```bash
# Aktivasi venv dulu
.venv\Scripts\activate

# Jalankan AI server (default port 8000)
energym-server

# Atau dengan uvicorn langsung
uvicorn energym_ai.server:app --host 0.0.0.0 --port 8000
```

Server berjalan di `http://0.0.0.0:8000`. Mobile app akan connect ke IP laptop ini.

### Standalone Mode (tanpa mobile, tampil di layar laptop)

```bash
energym-run --exercise bicep_curl
energym-run --exercise alternating_curl
energym-run --exercise hammer_curl
```

---

## Generate QR Station

QR code digunakan mobile app untuk mengetahui IP dan station_id gym station.

```bash
# Generate QR baru (auto-detect IP laptop saat ini)
python scripts/generate_qr.py

# Custom station ID (auto-increment jika sudah ada di DB)
python scripts/generate_qr.py --station-id STATION_01

# Custom output file
python scripts/generate_qr.py --station-id STATION_01 --output qr_lantai2.png
```

**Logika auto-increment:** Jika `STATION_01` sudah ada di tabel `stations` Supabase, script otomatis mencoba `STATION_02`, `STATION_03`, dst. Station baru langsung di-INSERT ke database.

---

## API Endpoints

Base URL: `http://<IP_LAPTOP>:8000`

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Cek status server dan kamera |
| `POST` | `/camera/warmup` | Pre-load kamera sebelum sesi dimulai |
| `POST` | `/session/start` | Mulai sesi workout AI |
| `POST` | `/session/stop` | Hentikan sesi dan dapatkan summary |
| `GET` | `/session/{session_id}/status` | Status sesi yang sedang berjalan |
| `GET` | `/stream/snapshot` | Snapshot JPEG frame terbaru dari webcam |
| `WS` | `/ws/{session_id}` | WebSocket real-time frame update |

### `POST /session/start`

```json
{
  "user_id": "uuid-user",
  "station_id": "STATION_01",
  "exercise_id": "uuid-exercise",
  "exercise_name": "Bicep Curl",
  "workout_id": "uuid-workout"
}
```

Response:
```json
{ "session_id": "a1b2c3d4" }
```

### WebSocket `/ws/{session_id}` — Frame Update

```json
{
  "type": "frame_update",
  "rep_count": 5,
  "state": "up",
  "elbow_angle": 48.3,
  "is_bad_form": false,
  "form_issues": [],
  "active_arm": "right"
}
```

### WebSocket — Session Ended

```json
{
  "type": "session_ended",
  "valid_reps": 10,
  "bad_reps": 2,
  "accuracy": 83.3,
  "duration_seconds": 45
}
```

---

## Konfigurasi Exercise (`configs/default.yaml`)

```yaml
exercises:
  bicep_curl:
    angle_thresholds:
      down_position: 160   # Sudut siku saat posisi bawah (fully extended)
      up_position: 50      # Sudut siku saat posisi atas (fully curled)
    form_rules:
      max_body_sway: 15    # Maksimal deviasi badan (derajat)
      max_elbow_drift: 25  # Maksimal pergerakan siku ke depan (derajat)
      min_rom_angle: 90    # Minimum range of motion
    tempo:
      min_concentric: 0.8  # Waktu minimum fase angkat (detik)
      min_eccentric: 1.0   # Waktu minimum fase turun (detik)
```

---

## Database

Tabel yang ditulis oleh AI server:

| Tabel | Operasi | Keterangan |
|---|---|---|
| `workout_sessions` | INSERT | Hasil lengkap tiap sesi AI |
| `stations` | INSERT | Saat QR baru di-generate |

### Schema `workout_sessions`

```sql
user_id, workout_id, exercise_id, station_id,
exercise_name, duration_seconds,
valid_reps, bad_reps, accuracy, created_at
```

---

## Troubleshooting

**Server tidak bisa diakses dari HP**
- Pastikan laptop dan HP di WiFi yang sama
- Cek firewall Windows: izinkan port 8000
- Verifikasi IP: `ipconfig` lalu cari IPv4 Address

**Kamera tidak terdeteksi**
- Cek `camera.source` di `configs/default.yaml` (coba 0, 1, 2)
- Pastikan tidak ada aplikasi lain yang memakai webcam

**Supabase sync gagal**
- Cek `SUPABASE_URL` dan `SUPABASE_SERVICE_KEY` di `.env`
- Data tersimpan offline di `./data/offline_queue/` dan akan di-retry otomatis
