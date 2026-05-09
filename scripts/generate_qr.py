import json
import os
import re
import socket
import argparse


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _supabase_client():
    """Buat Supabase client dari .env atau environment variable."""
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[WARN] supabase / python-dotenv belum terinstall. Skip DB check.")
        return None

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        print("[WARN] SUPABASE_URL / SUPABASE_SERVICE_KEY tidak ditemukan. Skip DB check.")
        return None

    return create_client(url, key)


def resolve_station_id(client, requested_id: str) -> str:
    """
    Cek apakah station_code sudah ada di DB.
    Jika sudah ada, auto-increment suffix angkanya.
    Contoh: STATION_01 sudah ada → coba STATION_02, dst.
    Kembalikan station_id yang aman untuk dipakai.
    """
    if client is None:
        return requested_id

    try:
        resp = client.table("stations").select("station_code").execute()
        existing = {row["station_code"] for row in (resp.data or [])}
    except Exception as e:
        print(f"[WARN] Gagal query stations: {e}. Pakai ID yang diminta.")
        return requested_id

    if requested_id not in existing:
        return requested_id

    # Pisahkan prefix dan angka: "STATION_01" → prefix="STATION_", num=1
    match = re.match(r"^(.*?)(\d+)$", requested_id)
    if not match:
        # Tidak ada angka di akhir — tambahkan _02
        match = re.match(r"^(.*?)(\d+)$", requested_id + "01")
        prefix, num_str = requested_id + "_", "01"
    else:
        prefix, num_str = match.group(1), match.group(2)

    pad = len(num_str)
    num = int(num_str)

    while True:
        num += 1
        candidate = f"{prefix}{str(num).zfill(pad)}"
        if candidate not in existing:
            print(f"[INFO] '{requested_id}' sudah ada, pakai '{candidate}'")
            return candidate


def insert_station(client, station_code: str) -> None:
    """INSERT station baru ke tabel stations."""
    if client is None:
        return
    try:
        client.table("stations").insert({"station_code": station_code}).execute()
        print(f"[DB] Station '{station_code}' berhasil disimpan ke database.")
    except Exception as e:
        print(f"[WARN] Gagal INSERT station ke DB: {e}")


def generate(station_id: str, port: int, output: str) -> None:
    try:
        import qrcode
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Install dulu: pip install qrcode[pil] pillow")
        return

    # ── Cek DB & resolve station_id ──────────────────────────────────────
    client = _supabase_client()
    final_station_id = resolve_station_id(client, station_id)

    ip = get_local_ip()

    payload = json.dumps({
        "station_id": final_station_id,
        "ip": ip,
        "port": port,
    })

    print(f"IP yang terdeteksi : {ip}")
    print(f"Station ID         : {final_station_id}")
    print(f"QR Payload         : {payload}")

    # ── Generate QR ──────────────────────────────────────────────────────
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#FF6500", back_color="white")

    # Tambah label di bawah QR
    base = img.convert("RGB")
    w, h = base.size
    label_h = 60
    final = Image.new("RGB", (w, h + label_h), "white")
    final.paste(base, (0, 0))

    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    text = f"EnerGym Station: {final_station_id}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, h + 10), text, fill="#FF6500", font=font)
    draw.text(((w - text_w) // 2 + 10, h + 32), f"IP: {ip}:{port}", fill="#888", font=font)

    final.save(output)
    print(f"\nQR berhasil dibuat: {output}")

    # ── Simpan station ke DB setelah QR sukses dibuat ─────────────────────
    insert_station(client, final_station_id)

    print("Cetak file ini dan tempel di alat gym.")
    print("\nPERINGATAN: Pastikan HP dan laptop di WiFi yang SAMA saat latihan!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-id", default="STATION_01")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="qr_station.png")
    args = parser.parse_args()

    generate(args.station_id, args.port, args.output)
